# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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
#   https://pruners.github.io/flitK
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

from collections import namedtuple
from tempfile import NamedTemporaryFile
import argparse
import csv
import datetime
import heapq
import glob
import hashlib
import logging
import multiprocessing as mp
import os
import shutil
import sqlite3
import subprocess as subp
import sys

import flitconfig as conf
import flit_update
import flitutil as util
try:
    import flitelf as elf
except ImportError:
    elf = None


brief_description = 'Bisect compilation to identify problematic source code'

def hash_compilation(compiler, optl, switches):
    'Takes a compilation and returns a 10 digit hash string.'
    return hashlib.sha1((compiler + optl + switches).encode()).hexdigest()[:10]

def try_get_compiler_flags(compiler_binary, projconf):
    '''
    Try to get the compiler flags for the given binary for both compiling and
    linking.  This it tries to do from the project configuration (loaded from
    flit-config.toml).  If we are unable to find a matching compiler based on
    the binary given, then return two empty strings.

    @param compiler_binary: binary_name (to find in PATH) or path to compiler
        binary
    @param projconf: A dictionary loaded from flit-config.toml for the project.
        Specifically, the list of compilers will be used, i.e., the list found
        in projconf['compiler'], using fields 'binary', 'fixed_compile_flags',
        and 'fixed_link_flags'.
    @return two strings representing the 'fixed_compile_flags' and the
        'fixed_link_flags' of the matching compiler from the given projconf.
        If no matching compiler is found, then return two empty strings.

    >>> try_get_compiler_flags('', {})
    ('', '')

    >>> try_get_compiler_flags('emptyname', {})
    ('', '')

    >>> try_get_compiler_flags(
    ...     'nonmatching',
    ...     {'compiler': [{'binary': 'mycompiler', 'fixed_compile_flags': 'FF',
    ...                    'fixed_link_flags': 'LL'}]})
    ('', '')

    >>> try_get_compiler_flags(
    ...     'mycompiler',
    ...     {'compiler': [{'binary': 'mycompiler', 'fixed_compile_flags': 'FF',
    ...                    'fixed_link_flags': 'LL'}]})
    ('FF', 'LL')
    '''
    compiler_map = {}
    if 'compiler' in projconf:
        compiler_map = {x['binary']: x for x in projconf['compiler']}
        which_map = {shutil.which(k): v for k, v in compiler_map.items()
                     if shutil.which(k)}
    else:
        return '', ''

    which_binary = shutil.which(compiler_binary)
    if compiler_binary in compiler_map:
        compiler = compiler_map[compiler_binary]
    elif which_binary in which_map:
        compiler = which_map[which_binary]
    else:
        return '', ''
    return compiler['fixed_compile_flags'], compiler['fixed_link_flags']

def try_resolve_compiler_type(compiler_name, projconf):
    '''
    Try to resolve the compiler type from the given compiler name and the
    project configuration (loaded from flit-config.toml).  If we are unable to
    find the compiler type in the project configuration, then return 'misc' as
    the compiler type.

    @param compiler_name: binary name (to find in PATH) or path to compiler
        binary
    @param projconf: A dictionary loaded from flit-config.toml for the project.
        Specifically, the list of compilers will be used, i.e., the list found
        in projconf['compiler'], using fields 'binary' and 'type'
    @return the compiler type from ('clang', 'gcc', 'intel') or 'misc' if it
        cannot be determined.

    >>> try_resolve_compiler_type('', {})
    'misc'
    >>> try_resolve_compiler_type('emptyconfig', {})
    'misc'
    >>> try_resolve_compiler_type(
    ...     'mycompiler',
    ...     {'compiler': [{'binary': 'mycompiler', 'type': 'gcc'}]})
    'gcc'
    >>> try_resolve_compiler_type(
    ...     'mycompiler',
    ...     {'compiler': [{'binary': 'mycompiler', 'type': 'clang'}]})
    'clang'
    >>> try_resolve_compiler_type(
    ...     'path/to/mycompiler',
    ...     {'compiler': [
    ...         {'binary': 'mycompiler', 'type': 'gcc'},
    ...         {'binary': 'path/to/mycompiler', 'type': 'intel'},
    ...         ]})
    'intel'
    >>> try_resolve_compiler_type(
    ...     'wrong/path/to/mycompiler',
    ...     {'compiler': [
    ...         {'binary': 'mycompiler', 'type': 'gcc'},
    ...         {'binary': 'path/to/mycompiler', 'type': 'intel'},
    ...         ]})
    'misc'
    '''
    if compiler_name == '':
        return 'misc'
    if 'compiler' not in projconf:
        return 'misc'
    for compiler in projconf['compiler']:
        if 'type' not in compiler or 'binary' not in compiler:
            continue
        if compiler_name == compiler['binary']:
            return compiler['type']
        compiler_path = shutil.which(compiler_name)
        binary_path = shutil.which(compiler['binary'])
        if compiler_path is not None and compiler_path == binary_path:
            return compiler['type']
    # TODO: do we want to add logic to call the compiler's --version flag and
    # TODO- try to guess which compiler type it is?  Not yet.
    return 'misc'

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

def create_bisect_makefile(directory, replacements, gt_src,
                           trouble_src=tuple(),
                           split_symbol_map=None):
    '''
    Returns the name of the created Makefile within the given directory, having
    been populated with the replacements, gt_src, and trouble_src.  It is then
    ready to be executed by 'make bisect' from the top-level directory of the
    user's flit tests.

    @param directory: (str) path where to put the created Makefile
    @param replacements: (dict) key -> value.  The key is found in the
        Makefile_bisect_binary.in and replaced with the corresponding value.
    @param gt_src: (list) which source files would be compiled with the
        ground-truth compilation within the resulting binary.
    @param trouble_src: (list) which source files would be compiled with the
        trouble compilation within the resulting binary.
    @param split_symbol_map:
        (dict fname -> list [list baseline symbols, list differing symbols])
        Files to compile as a split between baseline and differing, specifying
        baseline and differing symbols for each file.

    Within replacements, there are some optional fields:
    - added_link_flags: (list) (optional) List of linker flags to give to the
          ground-truth compiler when performing linking.

    @return the bisect makefile name without directory prepended to it
    '''
    # TODO: refactor this function.  it is too complicated
    if split_symbol_map is None:
        split_symbol_map = {} # default to an empty dictionary
    repl_copy = dict(replacements)
    projdir = os.path.join(directory, '..')
    projconf = util.load_projconf(projdir)
    cxxflags, tbl_ldflags = try_get_compiler_flags(repl_copy['trouble_cxx'],
                                                   projconf)
    try:
        with util.pushd(projdir):
            tbl_ldflags += ' {}'.format(flit_update._additional_ldflags(
                {'type': repl_copy['trouble_type'],
                 'binary': repl_copy['trouble_cxx']}))
    except NotImplementedError:
        pass # skip over unsupported compiler types
    repl_copy['trouble_cxxflags'] = cxxflags
    repl_copy['trouble_ldflags'] = tbl_ldflags
    if repl_copy['bisect_ldflags'] is None:
        # TODO: have this be linker flags for $(BISECT_LINK) executable
        #   (similar to ldflags and cxxflags)
        if repl_copy['bisect_linker'] is not None:
            _, bisect_ldflags = try_get_compiler_flags(
                repl_copy['bisect_linker'], projconf)
            bisect_c_type = try_resolve_compiler_type(
                repl_copy['bisect_linker'], projconf)
            try:
                with util.pushd(projdir):
                    bisect_ldflags += ' {}'.format(
                        flit_update._additional_ldflags(
                            {'type': bisect_c_type,
                             'binary': repl_copy['bisect_linker']}))
            except NotImplementedError:
                pass # skip over unsupported compiler types
            repl_copy['bisect_ldflags'] = bisect_ldflags
        else:
            repl_copy['bisect_ldflags'] = '$(GT_LDFLAGS)'
    if repl_copy['bisect_linker'] is None:
        repl_copy['bisect_linker'] = '$(GT_CXX)'
    repl_copy['TROUBLE_SRC'] = '\n'.join(['TROUBLE_SRC      += {0}'.format(x)
                                          for x in trouble_src])
    repl_copy['BISECT_GT_SRC'] = '\n'.join(['BISECT_GT_SRC    += {0}'.format(x)
                                            for x in gt_src])
    repl_copy['SPLIT_SRC'] = '\n'.join(['SPLIT_SRC        += {0}'.format(x)
                                        for x in split_symbol_map])
    if 'added_link_flags' in repl_copy:
        repl_copy['EXTRA_LDFLAGS'] = '\n'.join([
            'LDFLAGS          += {0}'.format(x)
            for x in repl_copy['added_link_flags']])
        del repl_copy['added_link_flags']



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

    repl_copy['makefile'] = os.path.realpath(makepath)
    repl_copy['number'] = '{0:02d}'.format(num)
    logging.info('Creating makefile: %s', makepath)
    util.process_in_file(
        os.path.join(conf.data_dir, 'Makefile_bisect_binary.in'),
        makepath,
        repl_copy,
        overwrite=True)

    # Create the obj/symbols directory
    if len(split_symbol_map) > 0:
        for subdir in ('obj', os.path.join('obj', 'symbols')):
            try:
                os.mkdir(os.path.join(directory, subdir))
            except FileExistsError:
                pass # ignore if the directory already exists

    # Create the txt files containing symbol lists within the obj directory
    for split_srcfile, split_symbols in split_symbol_map.items():
        split_basename = os.path.basename(split_srcfile)
        split_base = os.path.join(directory, 'obj', 'symbols', split_basename)
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

def run_make(makefilename='Makefile', directory='.', verbose=False,
             jobs=mp.cpu_count(), target=None):
    '''
    Runs a make command.  If the build fails, then stdout and stderr will be
    output into the log.

    @param makefilename: Name of the Makefile (either absolute path or relative
        path to the given directory) (default 'Makefile')
    @param directory: Path to the directory to run in (default '.')
    @param verbose: True means echo the output to the console (default False)
    @param jobs: number of parallel jobs (default #cpus)
    @param target: Makefile target to build (defaults to GNU Make's default)

    @return None

    @throws subprocess.CalledProcessError on failure

    Configure the logger to spit to stdout instead of stderr
    >>> import logging
    >>> logger = logging.getLogger()
    >>> for handler in logger.handlers[:]:
    ...     logger.removeHandler(handler)
    >>> import sys
    >>> logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    Make sure the exception is thrown
    >>> with NamedTemporaryFile(mode='w') as tmpmakefile:
    ...     print('.PHONY: default\\n', file=tmpmakefile)
    ...     print('default:\\n', file=tmpmakefile)
    ...     print('\\t@echo hello\\n', file=tmpmakefile)
    ...     print('\\t@false\\n', file=tmpmakefile)
    ...     tmpmakefile.flush()
    ...
    ...     run_make(makefilename=tmpmakefile.name) # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    subprocess.CalledProcessError: Command ... returned non-zero exit status 2.

    See what was output to the logger
    >>> with NamedTemporaryFile(mode='w') as tmpmakefile:
    ...     print('.PHONY: default\\n', file=tmpmakefile)
    ...     print('default:\\n', file=tmpmakefile)
    ...     print('\\t@echo hello\\n', file=tmpmakefile)
    ...     print('\\t@false\\n', file=tmpmakefile)
    ...     tmpmakefile.flush()
    ...
    ...     try:
    ...         run_make(makefilename=tmpmakefile.name) #doctest: +ELLIPSIS
    ...     except:
    ...         pass
    ERROR:root:make error occurred.  Here is the output:
    make: Entering directory ...
    hello
    make: *** [...default] Error 1
    make: Leaving directory ...
    <BLANKLINE>

    Undo the logger configurations
    >>> for handler in logger.handlers[:]:
    ...     logger.removeHandler(handler)

    Create a temporary directory
    >>> import tempfile
    >>> temporary_directory = tempfile.mkdtemp()

    Given a directory and an absolute path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('.PHONY: default', file=makefile)
    ...     print('default:', file=makefile)
    ...     print('\\t@false', file=makefile)
    ...     print('.PHONY: correct', file=makefile)
    ...     print('correct:', file=makefile)
    ...     print('\\t@echo correct', file=makefile)
    ...     print('\\t@cat {}'.format(os.path.basename(makefilepath)),
    ...           file=makefile)
    ...     makefile.flush()
    ...     run_make(makefilename=makefilepath,
    ...              directory=temporary_directory,
    ...              target='correct') #doctest: +ELLIPSIS

    Given a directory and a relative path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('.PHONY: default', file=makefile)
    ...     print('default:', file=makefile)
    ...     print('\\t@false', file=makefile)
    ...     print('.PHONY: correct', file=makefile)
    ...     print('correct:', file=makefile)
    ...     print('\\t@echo correct', file=makefile)
    ...     print('\\t@cat {}'.format(os.path.basename(makefilepath)),
    ...           file=makefile)
    ...     makefile.flush()
    ...     curdir = os.path.realpath(os.curdir)
    ...     try:
    ...         os.chdir(os.path.dirname(temporary_directory))
    ...         run_make(os.path.basename(makefilepath),
    ...                  directory=os.path.basename(temporary_directory),
    ...                  target='correct') #doctest: +ELLIPSIS
    ...     except:
    ...         raise
    ...     finally:
    ...         os.chdir(curdir)

    Delete the temporary directory
    >>> import shutil
    >>> shutil.rmtree(temporary_directory)
    '''
    command = [
        'make',
        '-C', directory,
        '-f', makefilename,
        '-j', str(jobs),
        ]
    if target is not None:
        command.append(target)

    with NamedTemporaryFile() as tmpout:
        try:
            if not verbose:
                subp.check_call(command, stdout=tmpout, stderr=subp.STDOUT)
            else:
                ps = subp.Popen(command, stdout=subp.PIPE, stderr=subp.STDOUT)
                out = subp.check_output(['tee', tmpout.name], stdin=ps.stdout, universal_newlines=True)
                ps.communicate()
                
                if ps.returncode != 0:
                    raise subp.CalledProcessError(ps.returncode, command)
                
                # write to stdout; can't set stdout in subprocess as non-file streams throw exceptions
                if sys.stdout != sys.__stdout__:
                    _ = sys.stdout.write(out)
        except:
            tmpout.flush()
            with open(tmpout.name, 'r') as tmpin:
                msg = 'make error occurred.  Here is the output:\n' \
                      + tmpin.read()
                logging.error('%s', msg)
            raise

def build_bisect(makefilename, directory,
                 target='bisect',
                 verbose=False,
                 jobs=mp.cpu_count()):
    '''
    Creates the bisect executable by executing a parallel make.

    You may alternatively specify a different target than bisect, for example
    'bisect-clean' to specify to clean the unnecessary files for the build,
    'bisect-smallclean' to clean unnecessary things without needing to
    recompile for the next bisect step, or
    'distclean' to clean everything, including the generated makefile.

    @param makefilename: the filepath to the makefile relative to the given
        directory
    @param directory: where to execute make
    @param target: Makefile target to run
    @param verbose: False means block output from GNU make and running
    @param jobs: number of parallel jobs.  Defaults to #cpus

    @return None
    '''
    logging.info('Building the bisect executable')
    run_make(
        makefilename=makefilename,
        directory=directory,
        verbose=verbose,
        jobs=jobs,
        target=target)

def update_gt_results(directory, verbose=False,
                      jobs=mp.cpu_count(), fpic=False):
    '''
    Update the ground-truth.csv results file for FLiT tests within the given
    directory.

    @param directory: where to execute make
    @param verbose: False means block output from GNU make and running
    @param jobs: number of parallel jobs (default #cpus)
    @param fpic: True means compile the gt-fpic files too (default False)

    @return None
    '''
    gt_resultfile = util.extract_make_var('GT_OUT', directory=directory)[0]
    logging.info('Updating ground-truth results - %s', gt_resultfile)
    print('Updating ground-truth results -', gt_resultfile, end='', flush=True)
    run_make(
        directory=directory,
        verbose=verbose,
        jobs=jobs,
        target=gt_resultfile)
    if fpic:
        run_make(
            directory=directory,
            verbose=verbose,
            jobs=jobs,
            target='gt-fpic')
    print(' - done')
    logging.info('Finished Updating ground-truth results')

def get_comparison_result(resultfile):
    '''
    Returns the floating-point comparison value stored in the flit comparison
    csv file (resultfile).

    The result is pulled simply from the comparison column (first row).

    >>> from tempfile import NamedTemporaryFile as TFile
    >>> with TFile(mode='w', delete=False) as fout:
    ...     _ = fout.write('name,host,compiler,...,comparison,...\\n'
    ...                    'test,name,clang++,...,15.342,...\\n')
    ...     fname = fout.name
    >>> get_comparison_result(fname)
    15.342

    Make sure NULL values are handled appropriately
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    'NULL\\n')
    >>> print(get_comparison_result(fname))
    None

    Try out a value that is not a number and not NULL
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    'coconut\\n')
    >>> get_comparison_result(fname)
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'coconut'

    Delete the file now that we're done with it
    >>> import os
    >>> os.remove(fname)
    '''
    with open(resultfile, 'r') as fin:
        parser = csv.DictReader(fin)
        # should only have one row
        for row in parser:
            val = row['comparison']
            return float(val) if val != 'NULL' else None

def is_result_differing(resultfile):
    '''
    Returns True if the results from the resultfile is considered 'differing',
    meaning it is a different answer from the ground-truth.

    @param resultfile: path to the results csv file after comparison
    @return True if the result is different from ground-truth

    Try out a positive value
    >>> from tempfile import NamedTemporaryFile as TFile
    >>> with TFile(mode='w', delete=False) as fout:
    ...     _ = fout.write('name,host,compiler,...,comparison,...\\n'
    ...                    'test,name,clang++,...,15.342,...\\n')
    ...     fname = fout.name
    >>> is_result_differing(fname)
    True

    Try out a value that is less than zero
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    '-1e-34\\n')
    >>> is_result_differing(fname)
    True

    Try out a value that is identically zero
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    '0.0\\n')
    >>> is_result_differing(fname)
    False

    Make sure NULL values are handled appropriately
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    'NULL\\n')
    >>> is_result_differing(fname)
    Traceback (most recent call last):
        ...
    TypeError: float() argument must be a string or a number, not 'NoneType'

    Try out a value that is not a number and not NULL
    >>> with open(fname, 'w') as fout:
    ...     _ = fout.write('comparison\\n'
    ...                    'coconut\\n')
    >>> is_result_differing(fname)
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'coconut'

    Delete the file now that we're done with it
    >>> import os
    >>> os.remove(fname)
    '''
    return float(get_comparison_result(resultfile)) != 0.0

_extract_symbols_memos = {}
BisectSymbolTuple = namedtuple('BisectSymbolTuple',
                               'src, symbol, demangled, fname, lineno')
def extract_symbols(file_or_filelist, objdir):
    '''
    Extracts symbols for the given source file(s).  The corresponding object is
    assumed to be in the objdir with ".o" appended to the end of the filename.

    @param file_or_filelist: (str or list(str)) source file(s) for which to get
        symbols.
    @param objdir: (str) directory where object files are compiled for the
        given files.

    @return two lists of SymbolTuple objects (funcsyms, remaining).
        The first is the list of exported functions that are strong symbols and
        have a filename and line number where they are defined.  The second is
        all remaining symbols that are strong, exported, and defined.
    '''

    # if it is not a string, then assume it is a list of strings
    if not isinstance(file_or_filelist, str):
        funcsym_tuples = []
        remainingsym_tuples = []
        for fname in file_or_filelist:
            funcsyms, remaining = extract_symbols(fname, objdir)
            funcsym_tuples.extend(funcsyms)
            remainingsym_tuples.extend(remaining)
        return (funcsym_tuples, remainingsym_tuples)

    # now we know it is a string, so assume it is a filename
    fname = file_or_filelist
    fbase = os.path.basename(fname)
    fobj = os.path.join(objdir, fbase + '.o')

    if fobj in _extract_symbols_memos:
        return _extract_symbols_memos[fobj]

    func_symbols, remaining_symbols = elf.extract_symbols(fobj)
    func_symbols = [
        BisectSymbolTuple(fname, sym.symbol, sym.demangled, sym.fname,
                          sym.lineno)
        for sym in func_symbols]
    remaining_symbols = [
        BisectSymbolTuple(fname, sym.symbol, sym.demangled, sym.fname,
                          sym.lineno)
        for sym in remaining_symbols]
    _extract_symbols_memos[fobj] = (func_symbols, remaining_symbols)
    return _extract_symbols_memos[fobj]

def memoize_strlist_func(func):
    '''
    Memoize a function that takes a list of strings and returns a value.  This
    function returns the memoized version.  It is expected that the list
    passed in will be in the same order.  This memoization will not
    work if for instance the input is first shuffled.

    >>> def to_memoize(strlist):
    ...     print(strlist)
    ...     return strlist[0]
    >>> memoized = memoize_strlist_func(to_memoize)
    >>> memoized(['a', 'b', 'c'])
    ['a', 'b', 'c']
    'a'
    >>> memoized(['a', 'b', 'c'])
    'a'
    >>> memoized(['e', 'a'])
    ['e', 'a']
    'e'
    >>> memoized(['a', 'b', 'c'])
    'a'
    >>> memoized(['e', 'a'])
    'e'
    '''
    memo = {}
    def memoized_func(strlist):
        'func but memoized'
        idx = tuple(sorted(strlist))
        if idx in memo:
            return memo[idx]
        value = func(strlist)
        memo[idx] = value
        return value
    return memoized_func

def bisect_biggest(score_func, elements, found_callback=None, k=1,
                   skip_verification=False):
    '''
    Performs the bisect search, attempting to find the biggest offenders.  This
    is different from bisect_search() in that this function only tries to
    identify the top k offenders, not all of them.  If k is greater than or
    equal to the total number of offenders, then bisect_biggest() is more
    expensive than bisect_search().

    We do not want to call score_func() very much.  We assume the score_func()
    is is an expensive operation.  We could go throught the list one at a time,
    but that would cause us to potentially call score_func() more than
    necessary.

    Note: The same assumption as bisect_search() is in place.  That is that all
    differing elements are independent.  This means if an element contributes
    to a differing score, then it would contribute to a differing score by
    itself as well.  This is not always true, and this function does not verify
    this assumption.  Instead, it will only return the largest singleton
    offenders.

    @param score_func: a function that takes one argument (to_test) and returns
        a number greater than zero if one of the elements in to_test causes the
        result to be differing.  This value returned is used to compare the
        elements so that the largest k differing elements are found and
        returned.  If all differing elements return the same numerical value,
        then this will be less efficient than bisect_search.
        Note: if the set of elements is not differing, then either return 0 or
        a negative value.
    @param elements: the elements to search over.  Subsets of this list will be
        given to score_func().
    @param found_callback: a callback function to be called on every found
        differing element.  Will be given two arguments, the element, and the
        score from score_func().
    @param k: number of biggest elements to return.  The default is to return
        the one biggest offender.  If there are less than k elements that
        return positive scores, then only the found offenders will be returned.
    @param skip_verification: skip the verification assertions for performance
        reasons.

    @return list of the biggest offenders with their scores
        [(elem, score), ...]

    >>> def score_func(x):
    ...     print('scoring:', x)
    ...     return -2*min(x) if x else 0

    >>> bisect_biggest(score_func, [1, 3, 4, 5, -1, 10, 0, -15, 3], k=3,
    ...                skip_verification=True)
    scoring: [1, 3, 4, 5, -1, 10, 0, -15, 3]
    scoring: [1, 3, 4, 5]
    scoring: [-1, 10, 0, -15, 3]
    scoring: [-1, 10]
    scoring: [0, -15, 3]
    scoring: [0]
    scoring: [-15, 3]
    scoring: [-15]
    scoring: [3]
    scoring: [-1]
    scoring: [10]
    [(-15, 30), (-1, 2)]

    >>> bisect_biggest(score_func, [-1, -2, -3, -4, -5], k=3,
    ...                found_callback=print)
    scoring: [-1, -2, -3, -4, -5]
    scoring: [-1, -2]
    scoring: [-3, -4, -5]
    scoring: [-3]
    scoring: [-4, -5]
    scoring: [-4]
    scoring: [-5]
    -5 10
    -4 8
    -3 6
    [(-5, 10), (-4, 8), (-3, 6)]

    >>> bisect_biggest(score_func, [])
    []

    >>> bisect_biggest(score_func, [-1, -2, -3, -4, -5], k=1)
    scoring: [-1, -2, -3, -4, -5]
    scoring: [-1, -2]
    scoring: [-3, -4, -5]
    scoring: [-3]
    scoring: [-4, -5]
    scoring: [-4]
    scoring: [-5]
    [(-5, 10)]

    Test that verification is performed
    >>> def score_func(x):
    ...     return -2*min(x) if len(x) > 1 else 0
    >>> bisect_biggest(score_func, [-1, -2], k=2)
    Traceback (most recent call last):
        ...
    AssertionError: Assumption that differing elements are independent was wrong

    >>> def score_func(x):
    ...     score = 0
    ...     if len(x) == 0: return 0
    ...     if min(x) < -1: score += max(0, -2*min(x))
    ...     if len(x) > 1: score += max(0, -2*max(x))
    ...     return score
    >>> bisect_biggest(score_func, [-1, -2], k=2)
    Traceback (most recent call last):
        ...
    AssertionError: Assumption that minimal sets are non-overlapping was wrong
    '''
    if len(elements) == 0:
        return []
    found_list = []
    frontier = []
    push = lambda x: heapq.heappush(frontier, (-score_func(x), x))
    pop = lambda: heapq.heappop(frontier)
    push(elements)
    while len(frontier) > 0 and frontier[0][0] < 0 and len(found_list) < k:
        score, elems = pop()
        if len(elems) == 1:
            found_list.append((elems[0], -score))
            if found_callback is not None:
                found_callback(elems[0], -score)
        else:
            push(elems[:len(elems) // 2])
            push(elems[len(elems) // 2:])

    if not skip_verification:
        if len(frontier) == 0 or frontier[0][0] >= 0:
            # found everything, so do the traditional assertions
            non_differing_list = \
                list(set(elements).difference(x[0] for x in found_list))
            assert score_func(non_differing_list) <= 0, \
                'Assumption that differing elements are independent was wrong'
            assert score_func(elements) == \
                score_func([x[0] for x in found_list]), \
                'Assumption that minimal sets are non-overlapping was wrong'

    return found_list

def bisect_search(score_func, elements, found_callback=None,
                  skip_verification=False):
    '''
    Performs the bisect search, attempting to find all elements contributing to
    a positive score.  This score_func() function is intended to identify when
    there are "differing" elements by returning a positive score.  This is
    different from bisect_biggest() in that we find all offenders and can
    therefore do some optimization that bisect_biggest() cannot do.

    We do not want to call score_func() very much.  We assume the score_func()
    is an expensive operation.  We could go throught the list one at a time,
    but that would cause us to potentially call score_func() more than
    necessary.

    This function has complexity
      O(k*log(n))*O(score_func)
    where n is the size of the elements and k is the number of differing
    elements to find.

    Note: A key assumption to this algorithm is that all differing elements are
    independent.  That may not always be true, so there are redundant checks
    within the algorithm to verify that this assumption is not vialoated.  If
    the assumption is found to be violated, then an AssertionError is raised.

    @param score_func: a function that takes one argument, the list of elements
        to test if they are differing.  The function then returns a positive
        value if the given list has a differing element.  If the given list
        does not have a differing element, this can return zero or a negative
        value.  Note: this function must be able to handle empty lists.  An
        empty list should instantly return a non-positive value, as there
        cannot possibly be a differing element passed to it.
        It is expected that this function is memoized because it may be called
        more than once on the same input during the execution of this
        algorithm.
    @param elements: contains differing elements, but potentially non-differing
        elements too
    @param found_callback: a callback function to be called on every found
        differing element.  Will be given two arguments, the element, and the
        score from score_func().
    @param skip_verification: skip the verification assertions for performance
        reasons.  This will run the score_func() two fewer times.

    @return minimal differing list of all elements that cause score_func() to
        return positive values, along with their scores, sorted descending by
        score.
        [(elem, score), ...]

    Here's an example of finding all negative numbers in a list.  Not very
    useful for this particular task, but it is demonstrative of how to use it.
    >>> call_count = 0
    >>> memo = {}
    >>> def score_func(x):
    ...     idx = tuple(sorted(x))
    ...     if idx not in memo:
    ...         global call_count
    ...         call_count += 1
    ...         memo[idx] = -2*min(x) if x else 0
    ...     return memo[idx]
    >>> bisect_search(score_func, [1, 3, 4, 5, -1, 10, 0, -15, 3])
    [(-15, 30), (-1, 2)]

    as a rough performance metric, we want to be sure our call count remains
    low for the score_func() function.  Note, we implemented memoization in
    score_func(), so we are only counting unique calls and not duplicate calls
    to score_func().
    >>> call_count
    10

    Test out the found_callback() functionality.
    >>> s = set()
    >>> bisect_search(score_func, [-1, -2, -3, -4],
    ...               found_callback=lambda x, y: s.add(x))
    [(-4, 8), (-3, 6), (-2, 4), (-1, 2)]
    >>> sorted(s)
    [-4, -3, -2, -1]

    See what happens when it has a pair that only show up together and not
    alone.  Only if -6 and 5 are in the list, then score_func() returns a
    positive value.  The assumption of this algorithm is that differing
    elements are independent, so this should throw an exception.
    >>> def score_func(x):
    ...     return max(x) - min(x) - 10 if x else 0
    >>> bisect_search(score_func, [-6, 2, 3, -3, -1, 0, 0, -5, 5])
    Traceback (most recent call last):
        ...
    AssertionError: Found element does not cause variability: 5

    Check that the assertion for found element is not turned off with
    skip_verification.
    >>> bisect_search(score_func, [-6, 2, 3, -3, -1, 0, 0, -5, 5],
    ...               skip_verification=True)
    Traceback (most recent call last):
        ...
    AssertionError: Found element does not cause variability: 5

    Check that the found_callback is not called on false positives.  Here I
    expect no output since no single element can be found.
    >>> try:
    ...     bisect_search(score_func, [-6, 2, 3, -3, -1, 0, 0, -5, 5],
    ...                   found_callback=print)
    ... except AssertionError:
    ...     pass

    Check that the verification can catch the other case of overlapping minimal
    sets.
    >>> def score_func(x):
    ...     score = 0
    ...     if len(x) == 0: return score
    ...     if min(x) < -5: score += 15 - min(x)
    ...     return score + max(x) - min(x) - 10
    >>> bisect_search(score_func, [-6, 2, 3, -3, -1, 0, -5, 5])
    Traceback (most recent call last):
        ...
    AssertionError: Assumption that minimal sets are non-overlapping was wrong

    Check that this is skipped with skip_verification
    >>> bisect_search(score_func, [-6, 2, 3, -3, -1, 0, -5, 5],
    ...               skip_verification=True)
    [(-6, 11)]
    '''
    if len(elements) == 0:
        return []

    # copy the incoming list so that we don't modify it
    quest_list = list(elements)

    differing_list = []
    while len(quest_list) > 0 and score_func(quest_list) > 0:

        # find one differing element
        quest_copy = quest_list
        while len(quest_copy) > 1:
            half_1 = quest_copy[:len(quest_copy) // 2]
            if score_func(half_1) > 0:
                quest_copy = half_1
            else:
                # optimization: mark half_1 as known, so that we don't need to
                # search it again
                quest_list = quest_list[len(half_1):]
                # update the local search
                quest_copy = quest_copy[len(half_1):]

        # since we remove known non-differing elements as we find them, the
        # differing element will be at the beginning of quest_list.
        differing_element = quest_list.pop(0)

        # double check that we found a differing element before declaring it
        # differing
        score = score_func([differing_element])
        assert score > 0, \
            'Found element does not cause variability: {}' \
            .format(differing_element)
        differing_list.append((differing_element, score))
        # inform caller that a differing element was found
        if found_callback is not None:
            found_callback(differing_element, score)

    if not skip_verification:
        # Perform a sanity check.  If we have found all of the differing items,
        # then compiling with all but these differing items will cause a
        # non-differing build.
        # This will fail if our hypothesis class is wrong
        non_differing_list = \
            list(set(elements).difference(x[0] for x in differing_list))
        assert score_func(non_differing_list) <= 0, \
            'Assumption that differing elements are independent was wrong'
        assert score_func(elements) == \
            score_func([x[0] for x in differing_list]),\
            'Assumption that minimal sets are non-overlapping was wrong'

    # sort descending by score
    differing_list.sort(key=lambda x: -x[1])

    return differing_list

def populate_parser(parser=None):
    '''
    Builds a parser, parses the arguments, and returns the parsed arguments.

    @param arguments: (list of str) arguments given to the program
    @param prog: (str) name of the program
    '''
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.description = '''
            Compiles the source code under both the ground-truth
            compilation and a given problematic compilation.  This tool
            then finds the minimal set of source files needed to be
            compiled under the problematic compilation flags so that the
            same answer is given.  This allows you to narrow down where the
            reproducibility problem lies.

            The log of the procedure will be kept in bisect.log.  Note that
            this file is overwritten if you call flit bisect again.
            '''

    # These positional arguments only make sense if not doing an auto run
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
                        # TODO: get the default test case to work
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
    parser.add_argument('--parallel', type=int, default=1,
                        help='''
                            How many parallel bisect searches to perform.  This
                            only makes sense with --auto-sqlite-run, since
                            there are multiple bisect runs to perform.  Each
                            bisect run is sequential.  This is distinct from
                            the --jobs argument.  This one specifies how many
                            instances of bisect to run, whereas --jobs
                            specifies how many compilation processes can be
                            spawned in parallel.
                            ''')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='''
                            Give verbose output including the output from the
                            Makefiles.  The default is to be quiet and to only
                            output short updates.
                            ''')
    processors = mp.cpu_count()
    parser.add_argument('-j', '--jobs', type=int, default=processors,
                        help='''
                            The number of parallel jobs to use for the call to
                            GNU make when performing the compilation.  Note,
                            this is not used when executing the tests, just in
                            compilation.
                            ''')
    parser.add_argument('-d', '--delete', action='store_true',
                        help='''
                            Automatically delete intermediate binaries and
                            output files.  This allows for much bigger
                            automatic runs where there is a concern for disk
                            space.  However, this option is not solely for the
                            --auto-sqlite-run option.  This will keep the
                            generated makefiles (e.g.
                            bisect-01/bisect-make-01.mk), the output
                            comparisons (e.g.
                            bisect-01/runbisect-01-out-comparison.csv), and the
                            log (e.g. bisect-01/bisect.log).  The things that
                            will not stay around are the executables (e.g.
                            bisect-01/runbisect-01), the saved output values
                            (e.g. runbisect-01-out_testcase_d.dat and
                            runbusect-01-out), or the object files (e.g.
                            bisect-01/obj/*).
                            ''')
    parser.add_argument('-k', '--biggest', metavar='K', type=int, default=None,
                        help='''
                            Instead of finding and returning all symbols that
                            cause variability, only return the largest K
                            contributors, with their contribution to
                            variability.  If K is close to the total number of
                            total contributing functions, then this is a much
                            slower approach in general than the full algorithm.
                            It is best if K is small.  This value used comes
                            from the custom comparison function you provide for
                            your flit test.

                            Note: many files (perhaps more than K, although
                            possibly less than K) may be found during the
                            search.  This is done to ensure that the symbols
                            identified are the largest contributors that are
                            identifiable through this approach.

                            Also note that using the --biggest flag, you
                            restrict the search space to only singleton
                            contributors.  That means that if there is a pair
                            of contributors that only contribute when they are
                            both compiled with the given compilation, then they
                            produce a measurable variation.
                            ''')
    parser.add_argument('--compile-only', action='store_true',
                        help='''
                            Only applicable with the --auto-sqlite-run option.
                            Only goes through the precompile step and then
                            exits.
                            ''')
    parser.add_argument('--precompile-fpic', action='store_true',
                        help='''
                            Only applicable with the --auto-sqlite-run option.
                            In the precompile phase, also precompiles the fPIC
                            object files into the top-level obj directory.
                            ''')
    parser.add_argument('--skip-verification', action='store_true',
                        help='''
                            By default, bisect will run some assertions
                            verifying that the assumptions made to have a
                            performant algorithm are valid.  This turns off
                            those assertions so as to not run the tests more
                            often than necessary.
                            ''')
    parser.add_argument('-t', '--compiler-type',
                        choices=('clang', 'gcc', 'intel', 'misc', 'auto'),
                        default='auto',
                        help='''
                            Specify the type of compiler used by the given
                            compilation.  The choices are 'clang', 'gcc',
                            'intel', 'misc', and auto.  The default is 'auto',
                            which will try to match the compiler with one
                            specified in flit-config.toml.  If not found there,
                            this tool falls back to 'misc', which assumes that
                            the user has ensured the correct flags are given in
                            the compilation to allow for a bisection search,
                            primarily that the standard c++ library used by the
                            given compiler is the same as that used by the
                            ground-truth compilation.
                            ''')
    parser.add_argument('--ldflags',
                        help='''
                            Replace the linker flags that would be used by the
                            linker program (pulled from 'fixed_link_flags' from
                            flit_config.toml).  If you want to add linker flags
                            instead of replacing them, use '--add-ldflags'
                            instead.

                            Note: since flags start with a '-', to not confuse
                            this argument parser, use the '=' syntax.  For
                            example: '--ldflags="-Wl,-rpath=/usr
                            -Wl,-rpath=/usr/lib64"'
                            ''')
    parser.add_argument('--add-ldflags', default='',
                        help='''
                            Add linker flags on top of the default that would be used.

                            Note: since flags start with a '-', to not confuse
                            this argument parser, use the '=' syntax.  For
                            example: '--add-ldflags="-Wl,-rpath=/usr
                            -Wl,-rpath=/usr/lib64"'
                            ''')
    parser.add_argument('--use-linker',
                        help='''
                            Specify a different linker to use.  This should
                            probably be a C++ compiler, like 'g++'.  If this
                            program is found in 'flit-config.toml', then the
                            linker flags used will be from the
                            'fixed_link_flags' specified there, unless
                            '--ldflags' is also specified.
                            ''')
    return parser

def parse_args(arguments, prog=None):
    '''
    Builds a parser, parses the arguments, and returns the parsed arguments.
    '''
    parser = populate_parser()
    if prog: parser.prog = prog
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

def test_makefile(args, makepath, testing_list, indent='  '):
    '''
    Runs the compilation in the makefile and returns the generated comparison
    result.

    @param args: parsed command-line arguments (see parse_args())
    @param makepath (str): absolute or relative path to the Makefile
    @param testing_list (list(str)): list of items being tested (for logging
        purposes)

    @return (float) generated comparison result
    '''
    print('{}Created {} - compiling and running'.format(indent, makepath),
          end='', flush=True)
    logging.info('%sCreated %s', indent, makepath)
    logging.info('%sChecking:', indent)
    for src in testing_list:
        logging.info('  %s%s', indent, src)

    relmakepath = os.path.relpath(makepath, args.directory)
    try:
        build_bisect(relmakepath, args.directory, verbose=args.verbose,
                     jobs=args.jobs)
    finally:
        if args.delete:
            build_bisect(relmakepath, args.directory, verbose=args.verbose,
                         jobs=args.jobs, target='bisect-smallclean')
    resultfile = util.extract_make_var('BISECT_RESULT', relmakepath,
                                       args.directory)[0]
    resultpath = os.path.join(args.directory, resultfile)
    result = get_comparison_result(resultpath)
    result_str = str(result)

    sys.stdout.write(' - score {0}\n'.format(result_str))
    logging.info('%sResult was %s', indent, result_str)

    return result

def _gen_bisect_lib_checker(args, bisect_path, replacements, sources,
                            indent='  '):
    '''
    Generates and returns the function that builds and check a list of
    libraries for showing variability.  The returned function is memoized, so
    no need to be careful to not call it more than once with the same
    arguments.
    '''
    def builder_and_checker(libs):
        '''
        Compiles all source files under the ground truth compilation and
        statically links in the libs.

        @param libs: static libraries to compile in

        @return The comparison value between this mixed compilation and the
            full baseline compilation.
        '''
        repl_copy = dict(replacements)
        repl_copy['added_link_flags'] = list(repl_copy['added_link_flags'])
        repl_copy['added_link_flags'].extend(libs)
        makefile = create_bisect_makefile(bisect_path, repl_copy, sources,
                                          [], dict())
        makepath = os.path.join(bisect_path, makefile)
        return test_makefile(args, makepath, libs, indent=indent)

    return memoize_strlist_func(builder_and_checker)

def _gen_bisect_source_checker(args, bisect_path, replacements, sources,
                               indent='  '):
    '''
    Generates and returns the function that builds and check a list of sources
    for showing variability.  The returned function is memoized, so no need to
    be careful to not call it more than once with the same arguments.
    '''
    def builder_and_checker(sources_to_optimize):
        '''
        Compiles the compilation with sources_to_optimize compiled with the
        optimized compilation and with everything else compiled with the
        baseline compilation.

        @param sources_to_optimize: source files to compile with the
            variability-inducing optimizations

        @return The comparison value between this mixed compilation and the
            full baseline compilation.
        '''
        gt_src = list(set(sources).difference(sources_to_optimize))
        makefile = create_bisect_makefile(bisect_path, replacements, gt_src,
                                          sources_to_optimize, dict())
        makepath = os.path.join(bisect_path, makefile)
        return test_makefile(args, makepath, sources_to_optimize, indent=indent)

    return memoize_strlist_func(builder_and_checker)

def _gen_bisect_symbol_checker(args, bisect_path, replacements, sources,
                               fsymbols, remainingsymbols, indent='  '):
    '''
    Generates and returns the function that builds and check a list of sources
    for showing variability.  The returned function is memoized, so no need to
    be careful to not call it more than once with the same arguments.
    '''
    def builder_and_checker(symbols_to_optimize):
        '''
        Compiles the compilation with all files compiled under the ground truth
        compilation except for the given symbols for the given files.

        In order to be able to isolate these symbols, the files will need to be
        compiled with -fPIC, but that is handled by the generated Makefile.

        @param symbols_to_optimize: (list of SymbolTuple) symbols to use
            from the variability-inducing optimization compilation

        @return The comparison value between this mixed compilation and the
            full baseline compilation.
        '''
        gt_symbols = list(set(fsymbols + remainingsymbols)
                          .difference(symbols_to_optimize))
        all_sources = list(sources)  # copy the list of all source files
        symbol_sources = [x.src for x in symbols_to_optimize + gt_symbols]
        trouble_src = []
        gt_src = list(set(all_sources).difference(symbol_sources))
        symbol_map = {x: [
            [y.symbol for y in gt_symbols if y.src == x],
            [z.symbol for z in symbols_to_optimize if z.src == x],
            ]
                      for x in symbol_sources}

        makefile = create_bisect_makefile(bisect_path, replacements, gt_src,
                                          trouble_src, symbol_map)
        makepath = os.path.join(bisect_path, makefile)
        symbol_strings = [
            '  {sym.fname}:{sym.lineno} {sym.symbol} -- {sym.demangled}'
            .format(sym=sym) for sym in symbols_to_optimize
            ]
        return test_makefile(args, makepath, symbol_strings, indent=indent)

    return memoize_strlist_func(builder_and_checker)

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
    memoized_checker = _gen_bisect_lib_checker(args, bisect_path, replacements,
                                               sources)

    print('Searching for differing intel static libraries:')
    logging.info('Searching for differing static libraries included by intel '
                 'linker:')
    #differing_library_msg = '    Found differing library {} (score {})'
    #differing_library_callback = lambda filename, score: \
    #    util.printlog(differing_library_msg.format(filename, score))
    #if args.biggest is None:
    #    differing_libs = bisect_search(
    #        memoized_checker, libs, found_callback=differing_library_callback,
    #        skip_verification=args.skip_verification)
    #else:
    #    differing_libs = bisect_biggest(
    #        memoized_checker, libs, found_callback=differing_library_callback,
    #        k=args.biggest, skip_verification=args.skip_verification)
    #return differing_libs
    if memoized_checker(libs):
        return libs
    return []

def search_for_source_problems(args, bisect_path, replacements, sources):
    '''
    Performs the search over the space of source files for problems.
    '''

    memoized_checker = _gen_bisect_source_checker(args, bisect_path,
                                                  replacements, sources)

    print('Searching for differing source files:')
    logging.info('Searching for differing source files under the trouble'
                 ' compilation')
    differing_source_msg = '    Found differing source file {}: score {}'
    differing_source_callback = lambda filename, score: \
        util.printlog(differing_source_msg.format(filename, score))
    differing_sources = bisect_search(memoized_checker, sources,
                                      found_callback=differing_source_callback,
                                      skip_verification=args.skip_verification)
    return differing_sources

def search_for_symbol_problems(args, bisect_path, replacements, sources,
                               differing_source, found_callback=None,
                               indent=''):
    '''
    Performs the search over the space of symbols within differing source files
    for problems.

    @param args: parsed command-line arguments
    @param bisect_path: directory where bisect is being performed
    @param replacements: dictionary of values to use in generating the Makefile
    @param sources: all source files
    @param differing_source: the one differing source file to search for
        differing symbols
    @param found_callback: (optional) a callback to be called on each found
        symbol
    @param indent: (optional) indentation to use for the logging and printing
        messages

    @return a list of identified differing symbols (if any), with their
        associated scores
        [(symbol, score), ...]
    '''
    print('{}Searching for differing symbols in: {}'.format(
        indent, differing_source))
    logging.info('%sSearching for differing symbols in: %s', differing_source,
                 indent)
    logging.info('%sNote: inlining disabled to isolate functions', indent)
    logging.info('%sNote: only searching over globally exported functions',
                 indent)
    logging.debug('%sSymbols:', indent)
    fsymbol_tuples, remaining_symbols = \
        extract_symbols(differing_source,
                        os.path.join(args.directory, 'obj', 'gt'))
    for sym in fsymbol_tuples:
        message = '{indent}  {sym.fname}:{sym.lineno} {sym.symbol} ' \
                  '-- {sym.demangled}'.format(indent=indent, sym=sym)
        logging.debug('%s', message)

    memoized_checker = _gen_bisect_symbol_checker(
        args, bisect_path, replacements, sources, fsymbol_tuples,
        remaining_symbols, indent=indent + '  ')

    # Check to see if -fPIC destroyed any chance of finding any differing
    # symbols
    if memoized_checker(fsymbol_tuples) <= 0.0:
        message_1 = '{}  Warning: -fPIC compilation destroyed the ' \
            'optimization'.format(indent)
        message_2 = '{}  Cannot find any trouble symbols'.format(indent)
        print(message_1)
        print(message_2)
        logging.warning('%s', message_1)
        logging.warning('%s', message_2)
        return []

    differing_symbol_msg = \
        '{indent}    Found differing symbol on line {sym.lineno} -- ' \
        '{sym.demangled} (score {score})'

    def differing_symbol_callback(sym, score):
        'Prints the finding and calls the registered callback'
        util.printlog(differing_symbol_msg.format(sym=sym, score=score,
                                                  indent=indent))
        if found_callback is not None:
            found_callback(sym, score)

    if args.biggest is None:
        differing_symbols = bisect_search(
            memoized_checker, fsymbol_tuples,
            found_callback=differing_symbol_callback,
            skip_verification=args.skip_verification)
    else:
        differing_symbols = bisect_biggest(
            memoized_checker, fsymbol_tuples,
            found_callback=differing_symbol_callback, k=args.biggest,
            skip_verification=args.skip_verification)
    return differing_symbols

def search_for_k_most_diff_symbols(args, bisect_path, replacements, sources):
    '''
    This function is similar to both search_for_source_problems() and
    search_for_symbol_problems().  This function will search for source
    problems AND also symbol problems such that the top k differing functions
    between the baseline compilation and the variability-inducing optimized
    compilation.

    @param args: parsed command-line arguments
    @param bisect_path: directory where bisect is being performed
    @param replacements: dictionary of values to use in generating the Makefile
    @param sources: all source files

    @return a list of identified differing symbols (if any), with their
        associated scores
        [(symbol, score), ...]
    '''
    assert args.biggest is not None
    assert args.biggest > 0

    util.printlog('Looking for the top {} different symbol(s) by starting with '
                  'files'.format(args.biggest))

    differing_symbols = []
    differing_source_msg = '    Found differing source file {}: score {}'
    differing_sources = []

    class ExitEarlyException(Exception):
        'Exception used to exit early from bisect search'
        pass

    def differing_symbol_callback(symbol, score):
        'captures the symbol and checks for early termination'
        assert len(differing_symbols) <= args.biggest
        assert score > 0

        if len(differing_symbols) >= args.biggest and \
                score < differing_symbols[-1][1]:
            # exit early because we're done with this file
            raise ExitEarlyException
        differing_symbols.append((symbol, score))
        differing_symbols.sort(key=lambda x: -x[1])
        differing_symbols[:] = differing_symbols[:args.biggest]

    symbol_search = lambda differing_source: \
        search_for_symbol_problems(
            args, bisect_path, replacements, sources, differing_source,
            found_callback=differing_symbol_callback, indent='    ')

    def differing_source_callback(filename, score):
        '''
        prints and captures the found source file, and checks for early
        termination.
        '''
        assert len(differing_symbols) <= args.biggest
        assert score > 0

        util.printlog(differing_source_msg.format(filename, score))
        differing_sources.append((filename, score))
        if len(differing_symbols) >= args.biggest and \
                score < differing_symbols[-1][1]:
            # exit early because we're done with this file
            raise ExitEarlyException

        try:
            symbol_search(filename)
        except ExitEarlyException:
            pass

    memoized_source_checker = _gen_bisect_source_checker(
        args, bisect_path, replacements, sources)
    try:
        # Note: ignore return because we already capture found sources
        bisect_biggest(memoized_source_checker, sources, k=len(sources),
                       found_callback=differing_source_callback)
    except ExitEarlyException:
        pass

    return differing_sources, differing_symbols

def gen_replacements(args, bisect_dir):
    'Build the replacements dictionary from parsed command-line arguments'
    hashval = hash_compilation(args.compiler, args.optl, args.switches)
    return {
        'bisect_dir': bisect_dir,
        'datetime': datetime.date.today().strftime("%B %d, %Y"),
        'flit_version': conf.version,
        'precision': args.precision,
        'test_case': args.testcase,
        'trouble_cxx': args.compiler,
        'trouble_optl': args.optl,
        'trouble_switches': args.switches,
        'trouble_type': args.compiler_type,
        'trouble_id': hashval,
        'bisect_ldflags': args.ldflags,
        'bisect_linker': args.use_linker,
        'added_link_flags': [args.add_ldflags],
        'build_gt_local': 'false',
        }

def compile_trouble(directory, replacements, verbose=False,
                    jobs=mp.cpu_count(), delete=True, fpic=False):
    '''
    Compiles the trouble executable for the given arguments.  This is useful to
    compile the trouble executable as it will force the creation of all needed
    object files for bisect.  This can be used to precompile all object files
    needed for bisect.
    '''
    # see if the Makefile needs to be regenerated
    # we use the Makefile to check for itself, sweet
    run_make(
        directory=directory,
        verbose=verbose,
        jobs=1,
        target='Makefile')

    # trouble compilations all happen in the same directory
    trouble_path = os.path.join(directory, 'bisect-precompile')
    try:
        os.mkdir(trouble_path)
    except FileExistsError:
        pass # not a problem if it already exists

    makefile = create_bisect_makefile(trouble_path, replacements, [])
    makepath = os.path.join(trouble_path, makefile)

    # Compile the trouble executable simply so that we have the object files
    build_bisect(makepath, directory, verbose=verbose,
                 jobs=jobs, target='trouble')
    if fpic:
        build_bisect(makepath, directory, verbose=verbose,
                     jobs=jobs, target='trouble-fpic')

    # Remove this prebuild temporary directory now
    if delete:
        shutil.rmtree(trouble_path)

def run_bisect(arguments, prog=None):
    '''
    The actual function for running the bisect command-line tool.

    Returns five things, (bisect_num, libs, sources, symbols, returncode).
    - bisect_num: (int) which bisect run this is.  Means the bisect results are
      stored in args.directory + '/bisect-' + str(bisect_num)
    - libs: (list of strings) problem libraries
    - sources: (list of strings) problem source files
    - symbols: (list of SymbolTuples) problem functions
    - returncode: (int) status, zero is success, nonzero is failure

    If the search fails in a certain part, then all subsequent items return
    None.  For example, if the search fails in the sources search, then the
    return value for sources and symbols are both None.  If the search fails in
    the symbols part, then only the symbols return value is None.
    '''
    args = parse_args(arguments, prog=prog)

    if args.compiler_type == 'auto':
        projconf = util.load_projconf(args.directory)
        args.compiler_type = try_resolve_compiler_type(args.compiler, projconf)

    # see if the Makefile needs to be regenerated
    # we use the Makefile to check for itself, sweet
    run_make(directory=args.directory, verbose=args.verbose, jobs=1,
             target='Makefile')

    # create a unique directory for this bisect run
    bisect_dir = create_bisect_dir(args.directory)
    bisect_num = int(bisect_dir.replace('bisect-', '').lstrip('0'))
    bisect_path = os.path.join(args.directory, bisect_dir)

    # keep a bisect.log of what was done, but need to remove all handlers,
    # otherwise logging.basicConfig() does nothing.
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(bisect_path, 'bisect.log'),
        filemode='w',
        format='%(asctime)s bisect: %(message)s',
        #level=logging.INFO)
        level=logging.DEBUG)

    replacements = gen_replacements(args, bisect_dir)

    logging.info('Starting the bisect procedure')
    logging.debug('  trouble compiler:           "%s"', args.compiler)
    logging.debug('  trouble optimization level: "%s"', args.optl)
    logging.debug('  trouble switches:           "%s"', args.switches)
    logging.debug('  trouble testcase:           "%s"', args.testcase)
    logging.debug('  trouble hash:               "%s"',
                  replacements['trouble_id'])

    # get the list of source files from the Makefile
    sources = util.extract_make_var('SOURCE', directory=args.directory)
    logging.debug('Sources')
    for source in sources:
        logging.debug('  %s', source)

    update_gt_results(args.directory, verbose=args.verbose, jobs=args.jobs)

    # Find out if the linker is to blame (e.g. intel linker linking mkl libs)
    differing_libs = []
    if os.path.basename(args.compiler) in ('icc', 'icpc'):
        warning_message = 'Warning: The intel compiler may not work with bisect'
        logging.info('%s', warning_message)
        print(warning_message)

        if '/' in args.compiler:
            compiler = os.path.realpath(args.compiler)
        else:
            compiler = os.path.realpath(shutil.which(args.compiler))

        # TODO: find a more portable way of finding the static libraries
        # TODO- This can be done by calling the linker command with -v to see
        # TODO- what intel uses in its linker.  The include path is in that
        # TODO- output command.
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
        try:
            differing_libs = search_for_linker_problems(
                args, bisect_path, replacements, sources, libs)
        except subp.CalledProcessError:
            print()
            print('  Executable failed to run.')
            print('Failed to search for differing libraries'
                  ' -- cannot continue.')
            return bisect_num, None, None, None, 1

        print('  differing static libraries:')
        logging.info('BAD STATIC LIBRARIES:')
        for lib in differing_libs:
            print('    ' + lib)
            logging.info('  %s', lib)
        if len(differing_libs) == 0:
            print('    None')
            logging.info('  None')

        # For now, if the linker was to blame, then say there may be nothing
        # else we can do.
        if len(differing_libs) > 0:
            message = 'May not be able to search further, because of intel ' \
                      'optimizations'
            print(message)
            logging.info('%s', message)

        # Compile all following executables with these static libraries
        # regardless of their effect
        replacements['added_link_flags'].extend(libs)

        # If the libraries were a problem, then reset what the baseline
        # ground-truth is, especially since we updated the LINK_FLAGS in the
        # generated Makefiles.
        if len(differing_libs) > 0:
            replacements['build_gt_local'] = 'true'

    try:
        if args.biggest is not None:
            differing_sources, differing_symbols = \
                search_for_k_most_diff_symbols(args, bisect_path,
                                               replacements, sources)
        else:
            differing_sources = search_for_source_problems(
                args, bisect_path, replacements, sources)
    except subp.CalledProcessError:
        print()
        print('  Executable failed to run.')
        print('Failed to search for differing sources -- cannot continue.')
        logging.exception('Failed to search for differing sources.')
        return bisect_num, differing_libs, None, None, 1

    if args.biggest is None:
        print('all variability inducing source file(s):')
        logging.info('ALL VARIABILITY INCUDING SOURCE FILE(S):')
    else:
        print('The found highest variability inducing source file{}:'.format(
            's' if len(differing_sources) > 1 else ''))
        logging.info('%d HIGHEST VARIABILITY SOURCE FILE%s:',
                     args.biggest, 'S' if args.biggest > 1 else '')

    for src in differing_sources:
        util.printlog('  {} (score {})'.format(src[0], src[1]))
    if len(differing_sources) == 0:
        util.printlog('  None')

    # Search for differing symbols one differing file at a time
    # This will allow us to maybe find some symbols where crashes before would
    # cause problems and no symbols would be identified
    #
    # Only do this if we didn't already perform the search above with
    #   search_for_k_most_diff_symbols()
    if args.biggest is None:
        differing_symbols = []
        for differing_source, _ in differing_sources:
            try:
                file_differing_symbols = search_for_symbol_problems(
                    args, bisect_path, replacements, sources, differing_source)
            except subp.CalledProcessError:
                print()
                print('  Executable failed to run.')
                print('Failed to search for differing symbols in {}'
                      '-- cannot continue'.format(differing_source))
                logging.exception(
                    'Failed to search for differing symbols in %s',
                    differing_source)
                return bisect_num, differing_libs, differing_sources, None, 1
            differing_symbols.extend(file_differing_symbols)
            if len(file_differing_symbols) > 0:
                if args.biggest is None:
                    message = '  All differing symbols in {}:'\
                              .format(differing_source)
                else:
                    message = '  {} differing symbol{} in {}:'.format(
                        args.biggest, 's' if args.biggest > 1 else '',
                        differing_source)
                print(message)
                logging.info(message)
                for sym, score in file_differing_symbols:
                    message = \
                        '    line {sym.lineno} -- {sym.demangled} ' \
                        '(score {score})'.format(sym=sym, score=score)
                    print(message)
                    logging.info('%s', message)

        if not args.skip_verification and len(differing_sources) > 1:
            # Verify that there are no missed files, i.e. those that are more
            # than singletons and that are to be grouped with one of the found
            # symbols.
            message = 'Verifying assumption about independent symbols'
            print(message)
            logging.info('%s', message)
            fsymbol_tuples, remaining_symbols = \
                extract_symbols([x[0] for x in differing_sources],
                                os.path.join(args.directory, 'obj', 'gt'))
            checker = _gen_bisect_symbol_checker(
                args, bisect_path, replacements, sources, fsymbol_tuples,
                remaining_symbols)
            assert checker(fsymbol_tuples) == \
                   checker([x[0] for x in differing_symbols]), \
                   'Assumption about independent symbols is False, ' \
                   'false negative results are possible'

    differing_symbols.sort(key=lambda x: (-x[1], x[0]))

    if args.biggest is None:
        print('All variability inducing symbols:')
        logging.info('ALL VARIABILITY INCUDING SYMBOLS:')
    else:
        print('The {} highest variability symbol{}:'
              .format(args.biggest, 's' if args.biggest > 1 else ''))
        logging.info('THE %d HIGHEST VARIABILITY INDUCING SYMBOL%s:',
                     args.biggest, 'S' if args.biggest > 1 else '')

    for sym, score in differing_symbols:
        message = \
            '  {sym.fname}:{sym.lineno} {sym.symbol} -- {sym.demangled} ' \
            '(score {score})'.format(sym=sym, score=score)
        print(message)
        logging.info('%s', message)
    if len(differing_symbols) == 0:
        print('    None')
        logging.info('  None')

    return bisect_num, differing_libs, differing_sources, differing_symbols, 0

def auto_bisect_worker(arg_queue, result_queue):
    '''
    Runs a worker that runs bisect and returns the obtained results into the
    result_queue.  Runs until the arg_queue is empty.

    @param arg_queue: (multiprocessing.Queue) queue contining lists of
        command-line arguments, each one associated with a single call to flit
        bisect.  This is where the work comes from.  These elements are tuples
        of
        - arguments: (list of str) base of command-line arguments minus
          positional
        - row: (dict of str -> val) The row to run from the sqlite database.
        - i: which job this is starting from i to rowcount
        - rowcount: total number of rows that exist
    @param result_queue: (multiprocessing.Queue or multiprocessing.SimpleQueue)
        queue for putting the results after running them.  This will contain a
        dictionary with the following keys:
        - compiler: (str) compiler used
        - optl: (str) optimization level
        - switches: (str) switches
        - libs: (list of str) differing libraries found
        - srcs: (list of str) differing source files found
        - syms: (list of SymbolTuple) differing symbols found
        - ret: (int) return code of running

    @return None
    '''
    import queue
    precision_map = {
        'f': 'float',
        'd': 'double',
        'e': 'long double',
        }
    try:
        while True:
            arguments, row, i, rowcount = arg_queue.get(False)

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
            print()
            print('Run', i, 'of', rowcount)
            print('flit bisect',
                  '--precision', precision,
                  '"' + compilation + '"',
                  testcase)

            num, libs, srcs, syms, ret = run_bisect(row_args)
            result_queue.put((row, num, libs, srcs, syms, ret))

    except queue.Empty:
        # exit the function
        pass

    except:
        # without putting something onto the result_queue, the parent process
        # will deadlock
        result_queue.put((row, -1, None, None, None, 1))
        raise

def parallel_auto_bisect(arguments, prog=None):
    '''
    Runs bisect in parallel under the auto mode.  This is only applicable if
    the --auto-sqlite-run option has been specified in the arguments.

    @return The sum of the return codes of each bisect call

    The results will be put into a file called auto-bisect.csv with the
    following columns:
    - testid:    The id of the row from the tests table in the sqlite db
    - compiler:  Compiler name from the db
    - optl:      Optimization level used, options are '-O0', '-O1', '-O2', and
                 '-O3'
    - switches:  Optimization flags used
    - precision: Precision checked, options are 'f', 'd', and 'e' for float,
                 double and extended respectively.
    - testcase:  FLiT test name run
    - type:      Type of result.  Choices are, 'completed', 'lib', 'src', and
                 'sim'.
        - completed:
                 comma-separated list of completed phases.  All phases are
                 'lib', 'src', and 'sym'.  This row is to help identify where
                 things errored out and failed to continue without parsing the
                 log files.
        - lib:   This row has a library reproducibility finding
        - src:   This row has a source file reproducibility finding
        - sym:   This row has a symbol reproducibility finding
    - name:      The value associated with the type from the type column.  For
                 completed, this has a comma-separated list of completed
                 phases.  For lib, the path to the blamed library.  For src,
                 the path to the blamed source file.  For sym, has the full
                 SymbolTuple() output string format, complete with source file,
                 line number, symbol name (before and after demangling), and
                 the filename where the symbol is located.

    Note: only for Intel compilations as the trouble compiler will the lib
    check actually be performed.  If the lib check is skipped, it will still
    show up in the list of completed steps, since it is considered to be a null
    step.
    '''
    # prepend a compilation and test case so that if the user provided
    # some, then an error will occur.
    args = parse_args(
        ['--precision', 'double', 'compilation', 'testcase'] + arguments,
        prog=prog)

    sqlitefile = args.auto_sqlite_run
    projconf = util.load_projconf(args.directory)

    try:
        connection = util.sqlite_open(sqlitefile)
    except sqlite3.DatabaseError:
        print('Error:', sqlitefile, 'is not an sqlite3 file')
        return 1

    query = connection.execute(
        'select * from tests where comparison != 0.0')
    rows = query.fetchall()
    precision_map = {
        'f': 'float',
        'd': 'double',
        'e': 'long double',
        }

    compilation_set = {(row['compiler'], row['optl'], row['switches'])
                       for row in rows}

    # see if the Makefile needs to be regenerated
    # we use the Makefile to check for itself, sweet
    run_make(directory=args.directory, verbose=args.verbose, jobs=1,
             target='Makefile')

    print('Before parallel bisect run, compile all object files')
    for i, compilation in enumerate(sorted(compilation_set)):
        args.compiler, args.optl, args.switches = compilation
        print('  ({0} of {1})'.format(i + 1, len(compilation_set)),
              ' '.join((args.compiler, args.optl, args.switches)) + ':',
              end='',
              flush=True)
        args.compiler_type = try_resolve_compiler_type(args.compiler, projconf)
        replacements = gen_replacements(args, 'bisect-precompile')
        compile_trouble(args.directory, replacements, verbose=args.verbose,
                        jobs=args.jobs, delete=args.delete,
                        fpic=args.precompile_fpic)
        print('  done', flush=True)

    # Update ground-truth results before launching workers
    update_gt_results(args.directory, verbose=args.verbose, jobs=args.jobs,
                      fpic=args.precompile_fpic)

    if args.compile_only:
        print('Done with precompilation -- exiting')
        return 0

    # Generate the worker queue
    arg_queue = mp.Queue()
    result_queue = mp.SimpleQueue()
    i = 0
    rowcount = len(rows)
    for row in rows:
        i += 1
        arg_queue.put((arguments, dict(row), i, rowcount))

    # Create the workers
    workers = []
    for _ in range(args.parallel):
        process = mp.Process(target=auto_bisect_worker,
                             args=(arg_queue, result_queue))
        process.start()
        workers.append(process)

    return_tot = 0
    with open('auto-bisect.csv', 'w') as resultsfile:
        writer = csv.writer(resultsfile)
        writer.writerow([
            'testid',
            'bisectnum',
            'compiler',
            'optl',
            'switches',
            'precision',
            'testcase',
            'type',
            'name',
            'return',
            ])
        resultsfile.flush()

        # Process the results
        for _ in range(rowcount):
            row, num, libs, srcs, syms, ret = result_queue.get()
            return_tot += ret

            entries = []
            completed = []
            if libs is not None:
                entries.extend([('lib', x) for x in libs])
                completed.append('lib')
            if srcs is not None:
                entries.extend([('src', x) for x in srcs])
                completed.append('src')
            if syms is not None:
                entries.extend([('sym', x) for x in syms])
                completed.append('sym')
            # prepend the completed items so it is first.
            entries = [('completed', ','.join(completed))] + entries

            for entry in entries:
                writer.writerow([
                    row['id'],
                    num,
                    row['compiler'],
                    row['optl'],
                    row['switches'],
                    precision_map[row['precision']],
                    row['name'],
                    entry[0],
                    entry[1],
                    ret,
                    ])
            resultsfile.flush()

    # Join the workers
    for process in workers:
        process.join()

    # Remove the files that were precompiled
    if args.delete:
        makepaths = glob.iglob(
            os.path.join(args.directory, 'bisect-*/bisect-make-01.mk'))
        for makepath in makepaths:
            makepath = os.path.relpath(makepath, args.directory)
            build_bisect(makepath, args.directory, verbose=args.verbose,
                         jobs=args.jobs, target='bisect-clean')

    return return_tot

def main(arguments, prog=None):
    '''
    A wrapper around the bisect program.  This checks for the --auto-sqlite-run
    stuff and runs the run_bisect multiple times if so.
    '''

    if elf is None:
        print('Error: binutils is not installed, bisect disabled',
              file=sys.stderr)
        return 1

    if '-a' in arguments or '--auto-sqlite-run' in arguments:
        return parallel_auto_bisect(arguments, prog)

    _, _, _, _, ret = run_bisect(arguments, prog)
    return ret

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
