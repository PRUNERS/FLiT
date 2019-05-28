# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Primarily copied from the scan-build package found at
#
#   https://github.com/rizsotto/scan-build.git
#
# That work on scan-build is under the LLVM Compiler Infrastructure and
# is distributed under the University of Illinois Open Source License.
#
# Modifications here were written by
#   Michael Bentley (mikebentley15@gmail.com),
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
Implements the autoconfig subcommand, taking a captured compilation database
and attempting to automatically generate a custom.mk configuration file.
'''

import argparse
import json
import logging
import os
import sys

import flitargformatter
import flitutil as util
import flitconfig as conf

brief_description = 'Autogenerates custom.mk from a JSON compilation database'

def parse_args(arguments, prog=sys.argv[0]):
    'Parse arguments and return parsed args'
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=flitargformatter.DefaultsParaSpaciousHelpFormatter,
        description='''
            Autogenerates a file used as a custom.mk file for FLiT.  It takes
            in a compilation database generated from "flit capture" and
            attempts to identify the necessary values for variables in
            custom.mk.

            By default, if the output file already exists (defaulted to
            "custom.mk", then the variables extracted from the compilation
            database will be appended to the file.  You may then move, add, or
            remove entries in the output file to suit your needs.

            Note: currently this tool is limited to generating the SOURCE and
            CC_REQUIRED variables in the "custom.mk", not the "LD_REQUIRED"
            values.
            ''',
        )
    parser.add_argument('-o', '--output', default='custom.mk',
                        help='''
                            Where to output the contents that would typically
                            be found in "custom.mk".
                            ''')
    parser.add_argument('--overwrite', action='store_true',
                        help='''
                            By default, if the output file already exists, we
                            append the generated values to the end.  But if
                            this flag is given, then the output file will be
                            overwritten.
                            ''')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debugging information')
    parser.add_argument('compilation_database',
                        help='Compilations database from "flit capture".')
    args = parser.parse_args(arguments)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    return args

def first(iterable, condition):
    '''
    Returns the first element from iterable such that condition(x) evaluates to
    True.  It is efficient because it exits early.

    >>> first(['a', 'b', 'c', 'b'], lambda x: x == 'b')
    'b'
    >>> first([], lambda x: False)
    >>> first([3, 2, 1, -1, -5], lambda x: x < 3)
    2
    '''
    for value in iterable:
        if condition(value):
            return value
    return None

def extract_cxxflags(arguments, command_cwd='.'):
    '''
    Extracts the command-line arguments that are c++ flags.

    @param arguments (list(str)) command-line arguments for a compilation.
    @param command_cwd (str) directory where command was given (for path
        evaulation)
    @return (list(str)) filtered out command-line arguments related to the C++
        compilation
    '''
    cxxflags = []
    # Map of ignored compiler option for the creation of a compilation database.
    #
    # Option names are mapped to the number of following arguments which should
    # be skipped.
    # TODO: skip more than these?
    flags_to_ignore = {
        '-c': 0,
        # preprocessor macros
        '-M': 0,
        '-MM': 0,
        '-MD': 0,
        '-MMD': 0,
        '-MG': 0,
        '-MP': 0,
        '-MF': 1,
        '-MT': 1,
        '-MQ': 1,
        # linker options
        '-static': 0,
        '-shared': 0,
        '-s': 0,
        '-rdynamic': 0,
        '-l': 1,
        '-L': 1,
        '-u': 1,
        '-z': 1,
        '-T': 1,
        '-Xlinker': 1,
        # clang-cl / msvc cl specific flags
        '-nologo': 0,
        '-EHsc': 0,
        '-EHa': 0,
        # output
        '-o': 1,
        }
    path_flags = [
        '-I',          # include
        '-isystem',    # include as system path
        '-iquote',     # include only for quoted includes
        '-idirafter',  # include after system includes
        '-include',    # include specified file
        '-imacros',    # include specified file for macros only
        ]
    define_flags = [
        '-D',          # define macro
        '-U',          # undefine macro
        ]
    # TODO: filter out optimization level
    # TODO: filter out flags that are specified in flit-config.toml
    args = iter(arguments)
    for arg in args:
        if not arg.startswith('-'):
            continue
        if arg in flags_to_ignore:
            count = flags_to_ignore[arg]
            for _ in range(count):
                next(args)
            continue
        # filter out link flags
        if any(arg.startswith(x) for x in ('-l', '-L', '-Wl,')):
            continue
        # filter out optimization level
        if arg.startswith('-O'):
            continue
        if any(arg.startswith(flag) for flag in path_flags):
            # allow for space separation and not space separation
            if arg in path_flags:
                flag = arg
                path = next(args)
            else:
                flag = first(path_flags, arg.startswith)
                path = arg[len(flag):]
            # relativize path, unless it is already absolute
            if not os.path.isabs(path):
                path = os.path.relpath(
                    os.path.join(command_cwd, path))
            cxxflags.append(flag + path)
        elif any(arg.startswith(flag) for flag in define_flags):
            # allow for space separation and not space separation
            if arg in define_flags:
                cxxflags.append(arg + next(args))
            else:
                cxxflags.append(arg)
        else:
            cxxflags.append(arg)
    return cxxflags

def extract_compilation_attributes(compilations):
    '''
    Extracts the attributes from the compilations needed for the custom.mk
    file.

    @param compilations (list(dict)) compilations taken from the compilation
        database file.
    @return (dict) fields needed for custom.mk
        - 'files': (list(str)) relative paths to compiled files
        - 'cxxflags': (list(str)) compilation flags

    >>> extract_compilation_attributes([])
    {'files': [], 'cxxflags': []}

    >>> attributes = extract_compilation_attributes([{
    ...     "arguments": [
    ...         "g++",
    ...         "-c",
    ...         "-g",
    ...         "-fPIC",
    ...         "-std=c++11",
    ...         "-Wall",
    ...         "-I.",
    ...         "-o",
    ...         "src/FlitCsv.o",
    ...         "src/FlitCsv.cpp"
    ...     ],
    ...     "compiler": "g++",
    ...     "directory": ".",
    ...     "file": "src/FlitCsv.cpp",
    ...     "language": "c++"
    ...     }])
    >>> attributes['files']
    ['src/FlitCsv.cpp']
    >>> attributes['cxxflags']
    ['-g', '-fPIC', '-std=c++11', '-Wall', '-I.']
    '''
    files = []
    cxxflags = None
    for compilation in compilations:
        logging.debug('compilation = %s', compilation)
        if compilation['language'] != 'c++':
            continue
        filepath = os.path.join(compilation['directory'], compilation['file'])
        files.append(os.path.relpath(filepath))
        if cxxflags is None:
            cxxflags = extract_cxxflags(compilation['arguments'],
                                        compilation['directory'])
            logging.debug('cxxflags = %s', cxxflags)
        else:
            this_cxxflags = extract_cxxflags(compilation['arguments'],
                                             compilation['directory'])
            if set(cxxflags) != set(this_cxxflags):
                print('Error: cxxflags mismatch')
                print('  cxxflags = {}'.format(cxxflags))
                print('  this_cxxflags = {}'.format(this_cxxflags))
                raise RuntimeError('cxxflags mismatch')
    logging.debug('files = %s', files)
    if cxxflags is None:
        cxxflags = []
    return {'files': files, 'cxxflags': cxxflags}

def gen_custom_makefile(outfile='custom.mk', files=None, cxxflags=None,
                        ldflags=None, overwrite=False):
    '''
    Generate the custom.mk file.

    @param outfile (str): filepath to output
    @param files (list(str)): list of source files
    @param cxxflags (list(str)): list of compiler flags
    @param ldflags (list(str)): list of linker flags
    @param overwrite (boolean): If true, will overwrite the file, else will
        append to the end of the existing file.
    @return None
    '''
    # handle default values
    files = files if files is not None else []
    cxxflags = cxxflags if cxxflags is not None else []
    ldflags = ldflags if ldflags is not None else []

    files_makevar = 'SOURCE'
    cxxflags_makevar = 'CC_REQUIRED'
    ldflags_makevar = 'LD_REQUIRED'
    # only append entries that are not there
    if os.path.isfile(outfile) and not overwrite:
        makevars = util.extract_make_vars(outfile)
        existing_files = set(makevars[files_makevar])
        existing_cxxflags = set(makevars[cxxflags_makevar])
        existing_ldflags = set(makevars[ldflags_makevar])

        # For debugging purposes, log what would be filtered out
        logging.debug('Files already found in %s: %s',
                      outfile, sorted(existing_files.intersection(files)))
        logging.debug('C++ flags already found in %s: %s',
                      outfile,
                      sorted(existing_cxxflags.intersection(cxxflags)))
        logging.debug('Link flags already found in %s: %s',
                      outfile, sorted(existing_ldflags.intersection(ldflags)))

        # We use list comprehension here instead of set subtraction so as to
        # preserve the original ordering, just filtered
        files = [f for f in files if f not in existing_files]
        cxxflags = [flag for flag in cxxflags if flag not in existing_cxxflags]
        ldflags = [flag for flag in ldflags if flag not in existing_ldflags]

    sources_str = ''
    if len(files) > 0:
        var_str = '{:14} += '.format(files_makevar)
        sources_str = var_str + ('\n' + var_str).join(files)
    cxxflags_str = ''
    if len(cxxflags) > 0:
        var_str = '{:14} += '.format(cxxflags_makevar)
        cxxflags_str = var_str + ('\n' + var_str).join(cxxflags)
    ldflags_str = ''
    if len(ldflags) > 0:
        var_str = '{:14} += '.format(ldflags_makevar)
        ldflags_str = var_str + ('\n' + var_str).join(ldflags)
    replacements = {
        'MORE_SOURCES': sources_str,
        'MORE_CXX_FLAGS': cxxflags_str,
        'MORE_LINK_FLAGS': ldflags_str,
        }

    if os.path.isfile(outfile) and not overwrite:
        print('Appending to {}'.format(outfile))
        keys = ('MORE_SOURCES', 'MORE_CXX_FLAGS', 'MORE_LINK_FLAGS')
        values = '\n\n'.join(replacements[key] for key in keys
                             if len(replacements[key]) > 0)
        if len(values) > 0:
            with open(outfile, 'a') as out:
                out.write('\n# Appending autogenerated by flit autoconfig\n')
                out.write(values + '\n')
        else:
            print('  Nothing to append.')
    else:
        if not os.path.exists(outfile):
            print('Creating {}'.format(outfile))
        else:
            print('Overwriting {}'.format(outfile))
        util.process_in_file(os.path.join(conf.data_dir, 'custom.mk.in'),
                             outfile, replacements, overwrite=True)

def main(arguments, prog=sys.argv[0]):
    'Main logic here'
    args = parse_args(arguments, prog)
    logging.debug('arguments: %s', args)

    with open(args.compilation_database, 'r') as infile:
        compilations = json.load(infile)
    attributes = extract_compilation_attributes(compilations)
    gen_custom_makefile(
        outfile=args.output,
        files=attributes['files'],
        cxxflags=attributes['cxxflags'],
        overwrite=args.overwrite,
        )

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
