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

from collections import namedtuple
import argparse
import csv
import datetime
import hashlib
import logging
import multiprocessing
import os
import re
import shutil
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

    Within replacements, there are some optional fields:
    - cpp_flags: (list) (optional) List of c++ compiler flags to give to
          each compiler when compiling object files from source files.
    - link_flags: (list) (optional) List of linker flags to give to the
          ground-truth compiler when performing linking.

    @return the bisect makefile name without directory prepended to it
    '''
    repl_copy = dict(replacements)
    repl_copy['TROUBLE_SRC'] = '\n'.join(['TROUBLE_SRC      += {0}'.format(x)
                                          for x in trouble_src])
    repl_copy['BISECT_GT_SRC'] = '\n'.join(['BISECT_GT_SRC    += {0}'.format(x)
                                            for x in gt_src])
    repl_copy['SPLIT_SRC'] = '\n'.join(['SPLIT_SRC        += {0}'.format(x)
                                        for x in split_symbol_map])
    if 'cpp_flags' in repl_copy:
        repl_copy['EXTRA_CC_FLAGS'] = '\n'.join(['CC_REQUIRED      += {0}'.format(x)
                                                 for x in repl_copy['cpp_flags']])
        del repl_copy['cpp_flags']
    if 'link_flags' in repl_copy:
        repl_copy['EXTRA_LD_FLAGS'] = '\n'.join(['LD_REQUIRED      += {0}'.format(x)
                                                 for x in repl_copy['link_flags']])
        del repl_copy['link_flags']


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
    sys.stdout.flush()
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

SymbolTuple = namedtuple('SymbolTuple', 'src, symbol, demangled, fname, lineno')
SymbolTuple.__doc__ = '''
Tuple containing information about the symbols in a file.  Has the following attributes:
    src:        source file that was compiled
    symbol:     mangled symbol in the compiled version
    demangled:  demangled version of symbol
    fname:      filename where the symbol is actually defined.  This usually
                will be equal to src, but may not be in some situations.
    lineno:     line number of definition within fname.
'''

def extract_symbols(file_or_filelist, objdir):
    '''
    Extracts symbols for the given file(s) given.  The corresponding object is
    assumed to be in the objdir with the filename replaced with the GNU Make
    pattern %.cpp=%_gt.o.
    
    @param file_or_filelist: (str or list(str)) source file(s) for which to get
        symbols.
    @param objdir: (str) directory where object files are compiled for the
        given files.

    @return a list of SymbolTuple objects
    '''
    symbol_tuples = []

    # if it is not a string, then assume it is a list of strings
    if not isinstance(file_or_filelist, str):
        for fname in file_or_filelist:
            symbol_tuples.extend(extract_symbols(fname, objdir))
        return symbol_tuples

    # now we know it is a string, so assume it is a filename
    fname = file_or_filelist
    fbase = os.path.splitext(os.path.basename(fname))[0]
    fobj = os.path.join(objdir, fbase + '_gt.o')

    # use nm and objdump to get the binary information we need
    symbol_strings = subp.check_output([
        'nm',
        '--extern-only',
        '--defined-only',
        fobj,
        ]).decode('utf-8').splitlines()
    demangled_symbol_strings = subp.check_output([
        'nm',
        '--extern-only',
        '--defined-only',
        '--demangle',
        fobj,
        ]).decode('utf-8').splitlines()
    objdump_strings = subp.check_output([
        'objdump', '--disassemble-all', '--line-numbers', fobj,
        ]).decode('utf-8').splitlines()

    # create the symbol -> (fname, lineno) map
    symbol_line_mapping = dict()
    symbol = None
    for line in objdump_strings:
        if len(line.strip()) == 0:        # skip empty lines
            continue
        if line[0].isdigit():             # we are at a symbol
            symbol = line.split()[1][1:-2]
            continue
        if symbol is None:                # if we don't have an active symbol
            continue                      # then skip
        srcmatch = re.search(':[0-9]+$', line)
        if srcmatch is not None:
            deffile = line[:srcmatch.start()]
            defline = int(line[srcmatch.start()+1:])
            symbol_line_mapping[symbol] = (deffile, defline)
            symbol = None                 # deactivate the symbol to not overwrite
            

    # generate the symbol tuples
    for i in range(len(symbol_strings)):
        symtype, symbol = symbol_strings[i].split(maxsplit=2)[1:3]
        demangled = demangled_symbol_strings[i].split(maxsplit=2)[2]
        try:
            deffile, defline = symbol_line_mapping[symbol]
        except KeyError:
            deffile, defline = None, None
        # We need to do all defined global symbols or we will get duplicate
        # symbol linker error
        #if symtype == 'T':
        symbol_tuples.append(
                SymbolTuple(fname, symbol, demangled, deffile, defline))

    return symbol_tuples

def bisect_search(is_bad, elements):
    '''
    Performs the bisect search, attempting to minimize the bad list.  We could
    go through the list one at a time, but that would cause us to call is_bad()
    more than necessary.  Here we assume that calling is_bad() is expensive, so
    we want to minimize calls to is_bad().  This function has
      O(k*log(n))*O(is_bad)
    where n is the size of the questionable_list and k is
    the number of bad elements in questionable_list.

    Note: A key assumption to this algorithm is that all bad elements are
    independent.  That may not always be true, so there are redundant checks
    within the algorithm to verify that this assumption is not vialoated.  If
    the assumption is found to be violated, then an AssertionError is raised.

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
    >>> x = bisect_search(is_bad, [1, 3, 4, 5, -1, 10, 0, -15, 3])
    >>> sorted(x)
    [-15, -1]

    as a rough performance metric, we want to be sure our call count remains
    low for the is_bad() function.
    >>> call_count
    9

    See what happens when it has a pair that only show up together and not
    alone.  Only if -6 and 5 are in the list, then is_bad returns true.
    The assumption of this algorithm is that bad elements are independent,
    so this should throw an exception.
    >>> def is_bad(x,y):
    ...     return max(x) - min(x) > 10
    >>> x = bisect_search(is_bad, [-6, 2, 3, -3, -1, 0, 0, -5, 5])
    Traceback (most recent call last):
        ...
    AssertionError: Assumption that bad elements are independent was wrong
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
                # TODO: possible optimization.
                #       if the length of Q2 is big enough, test
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

        # TODO: only double check if the last check was False
        # double check that we found a bad element before declaring it bad
        if is_bad([bad_element], known_list + quest_list):
            bad_list.append(bad_element)

        # add to the known list to not search it again
        known_list.append(bad_element)

    # Perform a sanity check.  If we have found all of the bad items, then
    # compiling with all but these bad items will cause a good build.
    # This will fail if our hypothesis class is wrong
    if len(bad_list) > 0:
        good_list = list(set(elements).difference(bad_list))
        assert not is_bad(good_list, bad_list), \
            'Assumption that bad elements are independent was wrong'

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
    parser.add_argument('-a', '--auto-sqlite-run', action='store',
                        required=False,
                        help='''
                            Automatically run bisect on all of the non-zero
                            comparison values in the given sqlite3 file.  If
                            you specify this option, then do not specify the
                            precision or the compilation or the testcase.
                            Those will be automatically procured from the
                            sqlite3 file.  The results will be stored in a csv
                            file called auto-bisect.csv.
                            ''')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='''
                            Give verbose output including the output from the
                            Makefiles.  The default is to be quiet and to only
                            output short updates.
                            ''')

    args = parser.parse_args(arguments)

    # Split the compilation into separate components
    split_compilation = args.compilation.strip().split(maxsplit=2)
    args.compiler = split_compilation[0]
    args.optl = ''
    args.switches = ''
    if len(split_compilation) > 1:
        args.optl = split_compilation[1]
    if len(split_compilation) > 2:
        args.switches = split_compilation[2]

    return args

def search_for_linker_problems(args, bisect_path, replacements, sources, libs):
    '''
    Performs the search over the space of statically linked libraries for
    problems.
    
    Linking will be done with the ground-truth compiler, but with the static
    libraries specified.  During this search, all source files will be compiled
    with the ground-truth compilation, but the static libraries will be
    included in the linking.

    Doing a binary search here actually breaks things since including some
    static libraries will require including others to resolve the symbols in
    the included static libraries.  So, instead this function just runs with
    the libraries included, and checks to see if there are reproducibility
    problems.
    '''
    def bisect_libs_build_and_check(trouble_libs, ignore_libs):
        '''
        Compiles all source files under the ground truth compilation and
        statically links in the trouble_libs.

        @param trouble_libs: static libraries to compile in
        @param ignore_libs: static libraries to ignore and not include
        
        @return True if the compilation has a non-zero comparison between this
            mixed compilation and the full ground-truth compilation.
        '''
        repl_copy = dict(replacements)
        repl_copy['link_flags'] = list(repl_copy['link_flags'])
        repl_copy['link_flags'].extend(trouble_libs)
        makefile = create_bisect_makefile(bisect_path, repl_copy, sources,
                                          [], dict())
        makepath = os.path.join(bisect_path, makefile)

        sys.stdout.write('  Create {0} - compiling and running' \
                         .format(makepath))
        sys.stdout.flush()
        logging.info('Created {0}'.format(makepath))
        logging.info('Checking:')
        for lib in trouble_libs:
            logging.info('  ' + lib)

        build_bisect(makepath, args.directory, verbose=args.verbose)
        resultfile = util.extract_make_var('BISECT_RESULT', makepath,
                                           args.directory)[0]
        resultpath = os.path.join(args.directory, resultfile)
        result_is_bad = is_result_bad(resultpath)

        result_str = 'bad' if result_is_bad else 'good'
        sys.stdout.write(' - {0}\n'.format(result_str))
        logging.info('Result was {0}'.format(result_str))

        return result_is_bad

    print('Searching for bad intel static libraries:')
    logging.info('Searching for bad static libraries included by intel linker:')
    #bad_libs = bisect_search(bisect_libs_build_and_check, libs)
    #return bad_libs
    if bisect_libs_build_and_check(libs, []):
        return libs
    else:
        return []

def search_for_source_problems(args, bisect_path, replacements, sources):
    '''
    Performs the search over the space of source files for problems.
    '''
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

    print('Searching for bad source files:')
    logging.info('Searching for bad source files under the trouble'
                 ' compilation')
    bad_sources = bisect_search(bisect_build_and_check, sources)
    return bad_sources

def search_for_symbol_problems(args, bisect_path, replacements, sources, bad_sources):
    '''
    Performs the search over the space of symbols within bad source files for
    problems.
    '''
    def bisect_symbol_build_and_check(trouble_symbols, gt_symbols):
        '''
        Compiles the compilation with all files compiled under the ground truth
        compilation except for the given symbols for the given files.

        In order to be able to isolate these symbols, the files will need to be
        compiled with -fPIC, but that is handled by the generated Makefile.

        @param trouble_symbols: (list of SymbolTuple) symbols to use
            from the trouble compilation
        @param gt_symbols: (list of SymbolTuple) symbols to use from
            the ground truth compilation

        @return True if the compilation has a non-zero comparison between this
            mixed compilation and the full ground truth compilation.
        '''
        all_sources = list(sources)  # copy the list of all source files
        symbol_sources = [x.src for x in trouble_symbols + gt_symbols]
        trouble_src = []
        gt_src = list(set(all_sources).difference(symbol_sources))
        symbol_map = { x: [
                            [y.symbol for y in gt_symbols if y.src == x],
                            [z.symbol for z in trouble_symbols if z.src == x],
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
        for sym in trouble_symbols:
            logging.info(
                    '  {sym.fname}:{sym.lineno} {sym.symbol} -- {sym.demangled}'
                    .format(sym=sym))

        build_bisect(makepath, args.directory, verbose=args.verbose)
        resultfile = util.extract_make_var('BISECT_RESULT', makepath,
                                           args.directory)[0]
        resultpath = os.path.join(args.directory, resultfile)
        result_is_bad = is_result_bad(resultpath)

        result_str = 'bad' if result_is_bad else 'good'
        sys.stdout.write(' - {0}\n'.format(result_str))
        logging.info('Result was {0}'.format(result_str))

        return result_is_bad

    print('Searching for bad symbols in the bad sources:')
    logging.info('Searching for bad symbols in the bad sources')
    logging.info('Note: inlining disabled to isolate functions')
    logging.info('Note: only searching over globally exported functions')
    logging.debug('Symbols:')
    symbol_tuples = extract_symbols(bad_sources, os.path.join(args.directory, 'obj'))
    for sym in symbol_tuples:
        message = '  {sym.fname}:{sym.lineno} {sym.symbol} -- {sym.demangled}' \
                  .format(sym=sym)
        logging.info(message)

    bad_symbols = bisect_search(bisect_symbol_build_and_check, symbol_tuples)
    return bad_symbols

def run_bisect(arguments, prog=sys.argv[0]):
    '''
    The actual function for running the bisect command-line tool.

    Returns four things, (libs, sources, symbols, returncode).
    - libs: (list of strings) problem libraries
    - sources: (list of strings) problem source files
    - symbols: (list of SymbolTuples) problem functions
    - returncode: (int) status, zero is success, nonzero is failure
    '''
    args = parse_args(arguments, prog)

    # our hash is the first 10 digits of a sha1 sum
    trouble_hash = hashlib.sha1(
        (args.compiler + args.optl + args.switches).encode()).hexdigest()[:10]

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
    logging.debug('  trouble compiler:           "{0}"'.format(args.compiler))
    logging.debug('  trouble optimization level: "{0}"'.format(args.optl))
    logging.debug('  trouble switches:           "{0}"'.format(args.switches))
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
        'trouble_cc': args.compiler,
        'trouble_optl': args.optl,
        'trouble_switches': args.switches,
        'trouble_id': trouble_hash,
        'link_flags': [],
        'cpp_flags': [],
        };

    update_gt_results(args.directory, verbose=args.verbose)

    # Find out if the linker is to blame (e.g. intel linker linking mkl libs)
    bad_libs = []
    if os.path.basename(args.compiler) in ('icc', 'icpc'):
        warning_message = 'Warning: The intel compiler may not work with bisect'
        logging.info(warning_message)
        print(warning_message)

        if '/' in args.compiler:
            compiler = os.path.realpath(args.compiler)
        else:
            compiler = os.path.realpath(shutil.which(args.compiler))

        # TODO: find a more portable way of finding the static libraries
        #       This can be done by calling the linker command with -v to see
        #       what intel uses in its linker.  The include path is in that
        #       output command.
        # Note: This is a hard-coded approach specifically for the intel linker
        #       and what I observed was the behavior.
        intel_dir = os.path.join(os.path.dirname(compiler), '..', '..')
        intel_dir = os.path.realpath(intel_dir)
        intel_lib_dir = os.path.join(intel_dir, 'compiler', 'lib', 'intel64')
        libs = [
            os.path.join(intel_lib_dir, 'libdecimal.a'),
            os.path.join(intel_lib_dir, 'libimf.a'),
            os.path.join(intel_lib_dir, 'libipgo.a'),
            os.path.join(intel_lib_dir, 'libirc_s.a'),
            os.path.join(intel_lib_dir, 'libirc.a'),
            os.path.join(intel_lib_dir, 'libirng.a'),
            os.path.join(intel_lib_dir, 'libsvml.a'),
            ]
        bad_libs = search_for_linker_problems(args, bisect_path, replacements,
                                              sources, libs)
        print('  bad static libraries:')
        logging.info('BAD STATIC LIBRARIES:')
        for lib in bad_libs:
            print('    ' + lib)
            logging.info('  ' + lib)
        if len(bad_libs) == 0:
            print('    None')
            logging.info('  None')

        #replacements['link_flags'].extend([
        #    '-L' + intel_lib_dir,
        #    '-lirc',
        #    '-lsvml',
        #    ])

        # TODO: If the linker is to blame, remove it from the equation for
        #       future searching This is done simply by using the ground-truth
        #       compiler to link instead of using the trouble compiler to link.

        # For now, if the linker was to blame, then exit saying there is
        # nothing else we can do.
        if len(bad_libs) > 0:
            message = 'May not be able to search further, because of intel'
            print(message)
            logging.info(message)

    # TODO: Handle the case where the ground-truth compiler is also an intel
    #       compiler.
    # TODO: Extra care must be taken when there is more than one intel linker
    #       specified.

    try:
        bad_sources = search_for_source_problems(args, bisect_path,
                                                 replacements, sources)
    except subp.CalledProcessError:
        print()
        print('  Executable failed to run.')
        print('Failed to search for bad sources -- cannot continue.')
        logging.exception('Failed to search for bad sources.')
        raise
        return bad_libs, [], [], 1

    print('  bad sources:')
    logging.info('BAD SOURCES:')
    for src in bad_sources:
        print('    ' + src)
        logging.info('  ' + src)
    if len(bad_sources) == 0:
        print('    None')
        logging.info('  None')


    try:
        bad_symbols = search_for_symbol_problems(args, bisect_path,
                                                 replacements, sources,
                                                 bad_sources)
    except subp.CalledProcessError:
        print()
        print('  Executable failed to run.')
        print('Failed to search for bad symbols -- cannot continue')
        logging.exception('Failed to search for bad symbols.')
        return bad_libs, bad_sources, [], 1

    print('  bad symbols:')
    logging.info('BAD SYMBOLS:')
    for sym in bad_symbols:
        message = '  {sym.fname}:{sym.lineno} {sym.symbol} -- {sym.demangled}' \
                  .format(sym=sym)
        print('  ' + message)
        logging.info(message)
    if len(bad_symbols) == 0:
        print('    None')
        logging.info('  None')

    return bad_libs, bad_sources, bad_symbols, 0

def main(arguments, prog=sys.argv[0]):
    '''
    A wrapper around the bisect program.  This checks for the --auto-sqlite-run
    stuff and runs the run_bisect multiple times if so.
    '''
    if '-a' in arguments or '--auto-sqlite-run' in arguments:
        if '-a' in arguments:
            idx = arguments.index('-a')
        else:
            idx = arguments.index('--auto-sqlite-run')

        arguments.pop(idx)
        sqlitefile = arguments.pop(idx)

        try:
            connection = util.sqlite_open(sqlitefile)
        except:
            print('Error:', sqlitefile, 'is not an sqlite3 file')
            return 1

        return_tot = 0
        with open('auto-bisect.csv', 'w') as resultsfile:
            writer = csv.writer(resultsfile)
            query = connection.execute(
                    'select * from tests where comparison_d!=0.0')
            precision_map = {
                'f': 'float',
                'd': 'double',
                'e': 'long double',
                }
            writer.writerow([
                'testid',
                'compiler',
                'precision',
                'testcase',
                'type',
                'name',
                'return',
                ])
            resultsfile.flush()
            entries = []
            for row in query:
                compilation = ' '.join(
                        [row['compiler'], row['optl'], row['switches']])
                testcase = row['name']
                precision = precision_map[row['precision']]
                row_args = list(arguments)
                row_args.extend([
                    '--precision', precision,
                    compilation,
                    testcase,
                    ])
                print('flit bisect',
                      '--precision', precision,
                      '"' + compilation + '"',
                      testcase)
                libs,srcs,syms,ret = run_bisect(row_args, prog)
                return_tot += ret

                entries.extend([('lib',x) for x in libs])
                entries.extend([('src',x) for x in srcs])
                entries.extend([('sym',x) for x in syms])

                for entry in entries:
                    writer.writerow([
                        row['id'],
                        compiler,
                        precision,
                        testcase,
                        entry[0],
                        entry[1],
                        ret,
                        ])
                resultsfile.flush()
        return return_tot

    else:
        _,_,_,ret = run_bisect(arguments, prog)
        return ret


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
