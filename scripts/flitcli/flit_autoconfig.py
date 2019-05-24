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
            ''',
        )
    # TODO: add --output
    # TODO: add --append
    # TODO: add --overwrite
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debugging information')
    parser.add_argument('compilation_database',
                        help='Compilations database from "flit capture".')
    args = parser.parse_args(arguments)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    return args

def extract_cxxflags(arguments):
    '''
    Extracts the command-line arguments that are c++ flags.

    @param arguments (list(str)) command-line arguments for a compilation.
    @return (list(str)) filtered out command-line arguments related to 
    '''
    cxxflags = []
    # Map of ignored compiler option for the creation of a compilation database.
    #
    # Option names are mapped to the number of following arguments which should
    # be skipped.
    flags_to_ignore = {
        '-c': 0,
        # preprocessor macros
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
    args = iter(arguments)
    for arg in args:
        if not arg.startswith('-'):
            continue
        if arg in flags_to_ignore:
            count = flags_to_ignore[arg]
            for _ in range(count):
                next(args)
            continue
        if arg == '-I':
            cxxflags.append(arg)
            cxxflags.append(next(args))
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
        logging.debug('compilation = {}'.format(compilation))
        if compilation['language'] != 'c++':
            continue
        filepath = os.path.join(compilation['directory'], compilation['file'])
        files.append(os.path.relpath(filepath))
        if cxxflags is None:
            cxxflags = extract_cxxflags(compilation['arguments'])
            logging.debug('cxxflags = {}'.format(cxxflags))
        else:
            assert set(cxxflags) == \
                set(extract_cxxflags(compilation['arguments']))
    logging.debug('files = {}'.format(files))
    if cxxflags is None:
        cxxflags = []
    return {'files': files, 'cxxflags': cxxflags}

def gen_custom_makefile(outfile='custom.mk', files=[], cxxflags=[], ldflags=[],
                        append=False):
    '''
    Generate the custom.mk file.

    @param outfile (str): filepath to output
    @param files (list(str)): list of source files
    @param cxxflags (list(str)): list of compiler flags
    @param ldflags (list(str)): list of linker flags
    @param append (boolean): If true, will append to the end of the existing
        file instead of overwriting it.
    @return None
    '''
    if append and not os.path.exists(outfile):
        append = False
    sources_str = ''
    if len(files) > 0:
        sources_str = 'SOURCE         += '
        sources_str += '\nSOURCE         += '.join(files)
    cxxflags_str = ''
    if len(cxxflags) > 0:
        cxxflags_str = 'CC_REQUIRED    += '
        cxxflags_str += '\nCC_REQUIRED    += '.join(cxxflags)
    linkflags_str = ''
    if len(ldflags) > 0:
        ldflags_str = 'LD_REQUIRED    += '
        ldflags_str += '\nLD_REQUIRED    += '.join(ldflags)
    replacements = {
        'MORE_SOURCES': sources_str,
        'MORE_CXX_FLAGS': cxxflags_str,
        'MORE_LINK_FLAGS': linkflags_str,
        }
    # TODO: print to screen what files are created/appended
    if append:
        print('Appending to {}'.format(outfile))
        with open(outfile, 'a') as out:
            out.write('# Appending autogenerated by flit autoconfig\n')
            out.write(replacements['MORE_SOURCES'] + '\n\n')
            out.write(replacements['MORE_CXX_FLAGS'] + '\n\n')
            out.write(replacements['MORE_LINK_FLAGS'] + '\n\n')
    else:
        print('Creating {}'.format(outfile))
        util.process_in_file(os.path.join(conf.data_dir, 'custom.mk.in'),
                             outfile, replacements, overwrite=True)

def main(arguments, prog=sys.argv[0]):
    'Main logic here'
    args = parse_args(arguments, prog)
    logging.debug('arguments: %s', args)
    exit_code = 0

    if os.path.isfile('custom.mk'):
        print('Error: custom.mk file already exists.  Do not want to' +
              ' overwrite',
              file=sys.stderr)
        return 1

    with open(args.compilation_database, 'r') as infile:
        compilations = json.load(infile)
    attributes = extract_compilation_attributes(compilations)
    gen_custom_makefile(
        files=attributes['files'],
        cxxflags=attributes['cxxflags'],
        append=True,
        )

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
