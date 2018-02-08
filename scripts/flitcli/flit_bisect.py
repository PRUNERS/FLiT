# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

'''
Implements the bisect subcommand, identifying the problematic subset of source
files that cause the variability.
'''

import flitconfig as conf
import flitutil as util

import toml

import argparse
import csv
import datetime
import hashlib
import logging
import multiprocessing
import os
import sqlite3
import subprocess as subp
import sys
import tempfile

brief_description = 'Bisect compilation to identify problematic source code'

def create_bisect_makefile(replacements, gt_src, trouble_src):
    '''
    Returns a tempfile.NamedTemporaryFile instance populated with the
    replacements, gt_src, and trouble_src.  It is then ready to be executed by
    'make bisect' from the correct directory.

    @param replacements: (dict) key -> value.  The key is found in the
        Makefile_bisect_binary.in and replaced with the corresponding value.
    @param gt_src: (list) which source files would be compiled with the
        ground-truth compilation within the resulting binary. 
    @param trouble_src: (list) which source files would be compiled with the
        trouble compilation within the resulting binary.

    @return (tempfile.NamedTemporaryFile) a temporary makefile that is
        populated to be able to compile the gt_src and the trouble_src into a
        single executable.
    '''
    repl_copy = dict(replacements)
    repl_copy['TROUBLE_SRC'] = '\n'.join(['TROUBLE_SRC      += {0}'.format(x)
                                          for x in trouble_src])
    repl_copy['BISECT_GT_SRC'] = '\n'.join(['BISECT_GT_SRC    += {0}'.format(x)
                                            for x in gt_src])
    # TODO: remove delete=False after done testing
    makefile = tempfile.NamedTemporaryFile(prefix='flit-bisect-', suffix='.mk',
                                           delete=False)
    repl_copy['Makefile'] = makefile.name
    logging.info('Creating makefile: ' + makefile.name)
    util.process_in_file(
        os.path.join(conf.data_dir, 'Makefile_bisect_binary.in'),
        makefile.name,
        repl_copy,
        overwrite=True)
    return makefile

def build_bisect(makefilename, directory, jobs=None):
    '''
    Creates the bisect executable by executing a parallel make.

    @param makefilename: the filepath to the makefile
    @param directory: where to execute make
    @param jobs: number of parallel jobs.  Defaults to #cpus

    @return None
    '''
    logging.info('Building the bisect executable')
    if jobs is None:
        jobs = multiprocessing.cpu_count()
    subp.check_call(
        ['make', '-C', directory, '-f', makefilename, '-j', str(jobs), 'bisect'],
        stdout=subp.DEVNULL, stderr=subp.DEVNULL)

def bisect_search(is_bad, elements):
    '''
    Performs the bisect search, attempting to minimize the bad list.  We could
    go through the list one at a time, but that would cause us to call is_bad()
    more than necessary.  Here we assume that calling is_bad() is expensive, so
    we want to minimize calls to is_bad().  This function has
      O(k*log(n))*O(is_bad)
    where n is the size of the questionable_list and k is
    the number of bad elements in questionable_list.

    @param is_bad: a function that takes two arguments (maybe_bad_list,
        maybe_good_list) and returns True if the maybe_bad_list has a bad
        element
    @param elements: contains bad elements, but potentially good elements too

    @return minimal bad list of all elements that cause is_bad() to return True

    Here's an example of finding all negative numbers in a list.  Not very
    useful for this particular task, but it is demonstrative of how to use it.
    >>> call_count = 0
    >>> def is_bad(x,y):
    ...     global call_count
    ...     call_count += 1
    ...     return min(x) < 0
    >>> x = bisect_search(is_bad, [1, 3, 4, 5, 10, -1, 0, -15])
    >>> sorted(x)
    [-15, -1]

    as a rough performance metric, we want to be sure our call count remains
    low for the is_bad() function.
    >>> call_count
    6
    '''
    # copy the incoming list so that we don't modify it
    quest_list = list(elements)
    known_list = []

    bad_list = []
    while len(quest_list) > 0 and is_bad(quest_list, known_list):

        # find one bad element
        Q = quest_list
        no_test = list(known_list)
        while len(Q) > 1:
            # split the questionable list into two lists
            Q1 = Q[:len(Q) // 2]
            Q2 = Q[len(Q) // 2:]
            if is_bad(Q1, no_test + Q2):
                Q = Q1
                no_test.extend(Q2)
                # TODO: if the length of Q2 is big enough, test
                #         is_bad(Q2, no_test + Q1)
                #       and if that returns False, then mark Q2 as known so
                #       that we don't need to search it again.
            else:
                # optimization: mark Q1 as known, so that we don't need to
                # search it again
                quest_list = quest_list[len(Q1):]
                known_list.extend(Q1)
                # update the local search
                Q = Q2
                no_test.extend(Q1)

        bad_element = quest_list.pop(0)
        bad_list.append(bad_element)
        known_list.append(bad_element)

        # double check that we found a bad element
        #assert is_bad([bad_element], known_list + quest_list)

    return bad_list

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Compiles the source code under both the ground-truth
                compilation and a given problematic compilation.  This tool
                then finds the minimal set of source files needed to be
                compiled under the problematic compilation flags so that the
                same answer is given.  This allows you to narrow down where the
                reproducibility problem lies.

                The log of the procedure will be kept in bisect.log.  Note that
                this file is overwritten if you call flit bisect again.
                ''',
            )
    parser.add_argument('compilation',
                        help='''
                            The problematic compilation to use.  This should
                            specify the compiler, optimization level, and
                            switches (which can be empty).  An example value
                            for this option would be "gcc -O2
                            -funsafe-math-optimizations" or
                            "/opt/intel/bin/icpc -O1".  The value will be split
                            into three groups using space separators, the first
                            is the compiler, the second is the optimization
                            level, and the third (if present) is the switches.
                            You will likely want to have this argument in
                            quotes since it will contain spaces.
                            ''')
    parser.add_argument('testcase',
                        help='''
                            The testcase to use.  You will need to specify one
                            of the tests.  You can find the list of test cases
                            by calling 'make dev' and then calling the created
                            executable './devrun --list-tests'.
                            ''')
                        # TODO: get the default case to work
                        #help='''
                        #    The testcase to use.  If there is only one test
                        #    case, then the default behavior is to use that test
                        #    case.  If there are more than one test case, then
                        #    you will need to specify one of them.  You can find
                        #    the list of test cases by calling 'make dev' and
                        #    then calling the created executable './devrun
                        #    --list-tests'.
                        #    ''')
    parser.add_argument('-C', '--directory', default='.',
                        help='The flit test directory to run the bisect tool')
    args = parser.parse_args(arguments)

    tomlfile = os.path.join(args.directory, 'flit-config.toml')
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        return 1

    # Split the compilation into the separate components
    split_compilation = args.compilation.strip().split(maxsplit=2)
    compiler = split_compilation[0]
    optl = ''
    switches = ''
    if len(split_compilation) > 1:
        optl = split_compilation[1]
    if len(split_compilation) > 2:
        switches = split_compilation[2]

    # our hash is the first 10 digits of a sha1 sum
    trouble_hash = hashlib.sha1(
        (compiler + optl + switches).encode()).hexdigest()[:10]

    # see if the Makefile needs to be regenerated
    # we use the Makefile to check for itself, sweet
    subp.check_call(['make', '-C', args.directory, 'Makefile'],
                    stdout=subp.DEVNULL, stderr=subp.DEVNULL)

    # keep a bisect.log of what was done
    logging.basicConfig(
        filename=os.path.join(args.directory, 'bisect.log'),
        filemode='w',
        format='%(asctime)s bisect: %(message)s',
        #level=logging.INFO)
        level=logging.DEBUG)

    logging.info('Starting the bisect procedure')
    logging.debug('  trouble compiler:           "{0}"'.format(compiler))
    logging.debug('  trouble optimization level: "{0}"'.format(optl))
    logging.debug('  trouble switches:           "{0}"'.format(switches))
    logging.debug('  trouble testcase:           "{0}"'.format(args.testcase))
    logging.debug('  trouble hash:               "{0}"'.format(trouble_hash))

    # get the list of source files from the Makefile
    sources = util.extract_make_var('SOURCE', 'Makefile',
                                    directory=args.directory)
    logging.debug('Sources')
    for source in sources:
        logging.debug('  ' + source)

    gt_src = sources[0:1]
    trouble_src = sources[1:]

    replacements = {
        'datetime': datetime.date.today().strftime("%B %d, %Y"),
        'flit_version': conf.version,
        'trouble_cc': compiler,
        'trouble_optl': optl,
        'trouble_switches': switches,
        'trouble_id': trouble_hash,
        };

    # TODO: what kind of feedback should we give the user while it is building?
    #       It is quite annoying as a user to simply issue a command and wait
    #       with no feedback for a long time.

    makefile = create_bisect_makefile(replacements, gt_src, trouble_src)
    build_bisect(makefile.name, args.directory)

    # TODO: should we delete these Makefiles?  I think so...
    #logging.info('Deleted makefile: ' + makefile.name)

    # TODO: determine if the problem is on the linker's side
    #       I'm not yet sure the best way to do that
    #       This is to be done later - first get for compilation problems

    # TODO: Perform (in parallel?) binary search from ground-truth and from
    #       problematic
    # - create extra Makefile
    # - run extra Makefile to determine if the problem is still there
    # - 

    # TODO: Use the custom comparison function in the test class when
    #       performing the binary search.

    # TODO: autogenerate Makefiles in the /tmp directly, preferrably with the
    #       tempfile module.


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
