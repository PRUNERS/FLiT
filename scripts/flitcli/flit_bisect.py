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

def create_bisect_dir(parent):
    '''
    Create a unique bisect directory named bisect-## where ## is the lowest
    integer that doesn't collide with an already existing file or directory.

    @param parent: the parent directory of where to create the bisect dir
    @return the bisect directory name without parent prepended to it

    >>> import tempfile
    >>> import shutil
    >>> import os

    >>> tmpdir = tempfile.mkdtemp()
    >>> create_bisect_dir(tmpdir)
    'bisect-01'
    >>> os.listdir(tmpdir)
    ['bisect-01']

    >>> create_bisect_dir(tmpdir)
    'bisect-02'
    >>> sorted(os.listdir(tmpdir))
    ['bisect-01', 'bisect-02']

    >>> create_bisect_dir(tmpdir)
    'bisect-03'
    >>> sorted(os.listdir(tmpdir))
    ['bisect-01', 'bisect-02', 'bisect-03']

    >>> create_bisect_dir(os.path.join(tmpdir, 'bisect-01'))
    'bisect-01'
    >>> sorted(os.listdir(tmpdir))
    ['bisect-01', 'bisect-02', 'bisect-03']
    >>> os.listdir(os.path.join(tmpdir, 'bisect-01'))
    ['bisect-01']

    >>> shutil.rmtree(tmpdir)
    '''
    num = 0
    while True:
        num += 1
        bisect_dir = 'bisect-{0:02d}'.format(num)
        try:
            os.mkdir(os.path.join(parent, bisect_dir))
        except FileExistsError:
            pass # repeat the while loop and try again
        else:
            return bisect_dir

def create_bisect_makefile(directory, replacements, gt_src, trouble_src,
                           split_symbol_map):
    '''
    Returns the name of the created Makefile within the given directory, having
    been populated with the replacements, gt_src, and trouble_src.  It is then
    ready to be executed by 'make bisect' from the top-level directory of the
    user's flit tests.

    @param directory: (str) path where to put the created Makefil
    @param replacements: (dict) key -> value.  The key is found in the
        Makefile_bisect_binary.in and replaced with the corresponding value.
    @param gt_src: (list) which source files would be compiled with the
        ground-truth compilation within the resulting binary.
    @param trouble_src: (list) which source files would be compiled with the
        trouble compilation within the resulting binary.
    @param split_symbol_map:
        (dict fname -> list [list good symbols, list bad symbols])
        Files to compile as a split between good and bad, specifying good and
        bad symbols for each file.

    @return the bisect makefile name without directory prepended to it
    '''
    repl_copy = dict(replacements)
    repl_copy['TROUBLE_SRC'] = '\n'.join(['TROUBLE_SRC      += {0}'.format(x)
                                          for x in trouble_src])
    repl_copy['BISECT_GT_SRC'] = '\n'.join(['BISECT_GT_SRC    += {0}'.format(x)
                                            for x in gt_src])
    repl_copy['SPLIT_SRC'] = '\n'.join(['SPLIT_SRC        += {0}'.format(x)
                                        for x in split_symbol_map])

    # Find the next unique file name available in directory
    num = 0
    while True:
        num += 1
        makefile = 'bisect-make-{0:02d}.mk'.format(num)
        makepath = os.path.join(directory, makefile)
        try:
            with open(makepath, 'x'):
                pass  # we just want to create the empty file
        except FileExistsError:
            pass
        else:
            break

    repl_copy['makefile'] = makepath
    repl_copy['number'] = '{0:02d}'.format(num)
    logging.info('Creating makefile: ' + makepath)
    util.process_in_file(
        os.path.join(conf.data_dir, 'Makefile_bisect_binary.in'),
        makepath,
        repl_copy,
        overwrite=True)

    # Create the obj directory
    if len(split_symbol_map) > 0:
        try:
            os.mkdir(os.path.join(directory, 'obj'))
        except FileExistsError:
            pass # ignore if the directory already exists

    # Create the txt files containing symbol lists within the obj directory
    for split_srcfile, split_symbols in split_symbol_map.items():
        split_basename = os.path.splitext(os.path.basename(split_srcfile))[0]
        split_base = os.path.join(directory, 'obj', split_basename)
        trouble_symbols_fname = split_base + '_trouble_symbols_' \
                + repl_copy['number'] + '.txt'
        gt_symbols_fname = split_base + '_gt_symbols_' \
                + repl_copy['number'] + '.txt'
        gt_symbols, trouble_symbols = split_symbols

        with open(gt_symbols_fname, 'w') as gt_fout:
            gt_fout.writelines('\n'.join(str(x) for x in gt_symbols))
        with open(trouble_symbols_fname, 'w') as trouble_fout:
            trouble_fout.writelines('\n'.join(str(x) for x in trouble_symbols))

    return makefile

def build_bisect(makefilename, directory, verbose=False, jobs=None):
    '''
    Creates the bisect executable by executing a parallel make.

    @param makefilename: the filepath to the makefile
    @param directory: where to execute make
    @param verbose: False means block output from GNU make and running
    @param jobs: number of parallel jobs.  Defaults to #cpus

    @return None
    '''
    logging.info('Building the bisect executable')
    if jobs is None:
        jobs = multiprocessing.cpu_count()
    kwargs = dict()
    if not verbose:
        kwargs['stdout'] = subp.DEVNULL
        kwargs['stderr'] = subp.DEVNULL
    subp.check_call(
        ['make', '-C', directory, '-f', makefilename, '-j', str(jobs), 'bisect'],
        **kwargs)

def update_gt_results(directory, verbose=False):
    '''
    Update the ground-truth.csv results file for FLiT tests within the given
    directory.

    @param directory: where to execute make
    @param verbose: False means block output from GNU make and running
    '''
    sys.stdout.flush()
    kwargs = dict()
    if not verbose:
        kwargs['stdout'] = subp.DEVNULL
        kwargs['stderr'] = subp.DEVNULL
    gt_resultfile = util.extract_make_var(
        'GT_OUT', os.path.join(directory, 'Makefile'))[0]
    logging.info('Updating ground-truth results - {0}'.format(gt_resultfile))
    print('Updating ground-truth results -', gt_resultfile, end='')
    subp.check_call(
        ['make', '-C', directory, gt_resultfile], **kwargs)
    print(' - done')
    logging.info('Finished Updating ground-truth results')

def is_result_bad(resultfile):
    '''
    Returns True if the results from the resultfile is considered 'bad',
    meaning it is a different answer from the ground-truth.

    @param resultfile: path to the results csv file after comparison
    @return True if the result is different from ground-truth
    '''
    with open(resultfile, 'r') as fin:
        parser = csv.DictReader(fin)
        # should only have one row
        for row in parser:
            # identical to ground truth means comparison is zero
            return float(row['comparison_d']) != 0.0

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

def parse_args(arguments, prog=sys.argv[0]):
    '''
    Builds a parser, parses the arguments, and returns the parsed arguments.

    @param arguments: (list of str) arguments given to the program
    @param prog: (str) name of the program
    '''
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
                        #    case, then the default behavior is to use that
                        #    test case.  If there are more than one test case,
                        #    then you will need to specify one of them.  You
                        #    can find the list of test cases by calling 'make
                        #    dev' and then calling the created executable
                        #    './devrun --list-tests'.
                        #    ''')
    parser.add_argument('-C', '--directory', default='.',
                        help='The flit test directory to run the bisect tool')
    parser.add_argument('-p', '--precision', action='store', required=True,
                        choices=['float', 'double', 'long double'],
                        help='''
                            Which precision to use for the test.  This is a
                            required argument, since it becomes difficult to
                            know when a difference happens with multiple
                            precision runs.  The choices are 'float', 'double',
                            and 'long double'.
                            ''')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='''
                            Give verbose output including the output from the
                            Makefiles.  The default is to be quiet and to only
                            output short updates.
                            ''')

    args = parser.parse_args(arguments)
    return args

def main(arguments, prog=sys.argv[0]):
    args = parse_args(arguments, prog)

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

    # create a unique directory for this bisect run
    bisect_dir = create_bisect_dir(args.directory)
    bisect_path = os.path.join(args.directory, bisect_dir)

    # keep a bisect.log of what was done
    logging.basicConfig(
        filename=os.path.join(bisect_path, 'bisect.log'),
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

    replacements = {
        'bisect_dir': bisect_dir,
        'datetime': datetime.date.today().strftime("%B %d, %Y"),
        'flit_version': conf.version,
        'precision': args.precision,
        'test_case': args.testcase,
        'trouble_cc': compiler,
        'trouble_optl': optl,
        'trouble_switches': switches,
        'trouble_id': trouble_hash,
        };

    # TODO: what kind of feedback should we give the user while it is building?
    #       It is quite annoying as a user to simply issue a command and wait
    #       with no feedback for a long time.

    def bisect_build_and_check(trouble_src, gt_src):
        '''
        Compiles the compilation with trouble_src compiled with the trouble
        compilation and with gt_src compiled with the ground truth compilation.

        @param trouble_src: source files to compile with trouble compilation
        @param gt_src: source files to compile with ground truth compilation

        @return True if the compilation has a non-zero comparison between this
            mixed compilation and the full ground truth compilation.
        '''
        makefile = create_bisect_makefile(bisect_path, replacements, gt_src,
                                          trouble_src, dict())
        makepath = os.path.join(bisect_path, makefile)

        sys.stdout.write('  Created {0} - compiling and running' \
                         .format(makepath))
        sys.stdout.flush()
        logging.info('Created {0}'.format(makepath))
        logging.info('Checking:')
        for src in trouble_src:
            logging.info('  ' + src)

        build_bisect(makepath, args.directory, verbose=args.verbose)
        resultfile = util.extract_make_var('BISECT_RESULT', makepath,
                                           args.directory)[0]
        resultpath = os.path.join(args.directory, resultfile)
        result_is_bad = is_result_bad(resultpath)

        result_str = 'bad' if result_is_bad else 'good'
        sys.stdout.write(' - {0}\n'.format(result_str))
        logging.info('Result was {0}'.format(result_str))

        return result_is_bad

    update_gt_results(args.directory, verbose=args.verbose)

    print('Searching for bad source files:')
    logging.info('Searching for bad source files under the trouble'
                 ' compilation')
    bad_sources = bisect_search(bisect_build_and_check, sources)
    print('  bad sources: ', bad_sources)

    symbol_tuples = []
    for bad_src in bad_sources:
        bad_base = os.path.splitext(os.path.basename(bad_src))[0]
        good_obj = os.path.join(args.directory, 'obj', bad_base + '_gt.o')
        symbol_strings = subp.check_output([
            'nm',
            '--extern-only',
            '--defined-only',
            good_obj,
            ]).splitlines()
        demangled_symbol_strings = subp.check_output([
            'nm',
            '--extern-only',
            '--defined-only',
            '--demangle',
            good_obj,
            ]).splitlines()
        for i in range(len(symbol_strings)):
            symbol = symbol_strings[i].split(maxsplit=2)[2].decode('utf-8')
            demangled = demangled_symbol_strings[i].split(maxsplit=2)[2] \
                        .decode('utf-8')
            symtype = symbol_strings[i].split()[1]
            # We need to do all defined global symbols or we will get duplicate
            # symbol linker error
            #if symtype == b'T':
            symbol_tuples.append((bad_src, symbol, demangled))

    print('Searching for bad symbols in the bad sources:')
    logging.info('Searching for bad symbols in the bad sources')
    logging.info('Note: inlining disabled to isolate functions')
    logging.info('Note: only searching over globally exported functions')
    logging.debug('Symbols:')
    for src, symbol, demangled in symbol_tuples:
        logging.debug('  {0}: {1} -- {2}'.format(src, symbol, demangled))

    def bisect_symbol_build_and_check(trouble_symbols, gt_symbols):
        '''
        Compiles the compilation with all files compiled under the ground truth
        compilation except for the given symbols for the given files.

        In order to be able to isolate these symbols, the files will need to be
        compiled with -fPIC, but that is handled by the generated Makefile.

        @param trouble_symbols: (tuple (src, symbol, demangled)) symbols to use
            from the trouble compilation
        @param gt_symbols: (tuple (src, symbol, demangled)) symbols to use from
            the ground truth compilation

        @return True if the compilation has a non-zero comparison between this
            mixed compilation and the full ground truth compilation.
        '''
        all_sources = list(sources)  # copy the list of all source files
        symbol_sources = [x[0] for x in trouble_symbols + gt_symbols]
        trouble_src = []
        gt_src = list(set(all_sources).difference(symbol_sources))
        symbol_map = { x: [
                            [y[1] for y in gt_symbols if y[0] == x],
                            [z[1] for z in trouble_symbols if z[0] == x],
                          ]
                       for x in symbol_sources }

        makefile = create_bisect_makefile(bisect_path, replacements, gt_src,
                                          trouble_src, symbol_map)
        makepath = os.path.join(bisect_path, makefile)

        sys.stdout.write('  Created {0} - compiling and running' \
                         .format(makepath))
        sys.stdout.flush()
        logging.info('Created {0}'.format(makepath))
        logging.info('Checking:')
        for src, symbol, demangled in trouble_symbols:
            logging.info('  {0}: {1} -- {2}'.format(src, symbol, demangled))

        build_bisect(makepath, args.directory, verbose=args.verbose)
        resultfile = util.extract_make_var('BISECT_RESULT', makepath,
                                           args.directory)[0]
        resultpath = os.path.join(args.directory, resultfile)
        result_is_bad = is_result_bad(resultpath)

        result_str = 'bad' if result_is_bad else 'good'
        sys.stdout.write(' - {0}\n'.format(result_str))
        logging.info('Result was {0}'.format(result_str))

        return result_is_bad

    bad_symbols = bisect_search(bisect_symbol_build_and_check, symbol_tuples)
    print('  bad symbols:')
    logging.info('BAD SYMBOLS:')
    for src, symbol, demangled in bad_symbols:
        print('    {0}: {1} -- {2}'.format(src, symbol, demangled))
        logging.info('    {0}: {1} -- {2}'.format(src, symbol, demangled))


    # TODO: determine if the problem is on the linker's side
    #       I'm not yet sure the best way to do that
    #       This is to be done later - first go for compilation problems


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
