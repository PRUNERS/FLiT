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
Implements the capture subcommand, capturing compilation process into a
database
'''

import argparse
import itertools
import json
import os
import sys

from libear import temporary_directory
from libscanbuild.compilation import (
    Compilation, CompilationDatabase, COMPILER_PATTERN_WRAPPER,
    COMPILER_PATTERNS_MPI_WRAPPER, COMPILER_PATTERNS_CC, COMPILER_PATTERNS_CXX
    )
from libscanbuild.intercept import (
    setup_environment, exec_trace_files, parse_exec_trace
    )
from libscanbuild import run_build, reconfigure_logging

brief_description = 'Captures source file compilations into a JSON database'

class CustomCompilation(Compilation):
    '''
    A compilation, but really a specialization of
    libscanbuild.compilation.Compilation for the kind of behaviour we need.
    '''

    def __init__(*args, **kwargs):
        'Just calls the Compilation class constructor'
        super().__init__(*args, **kwargs)

    @classmethod
    def _split_compiler(cls,        # type: Type[Compilation]
                        command,    # type: List[str]
                        cc,         # type: str
                        cxx         # type: str
                        ):
        '''
        Copied and modified from the Compilation class.

        @param cls (type) the current class
        @param command (list(str)) the command to classify
        @param cc (str) user specified C compiler name
        @param cxx (str) user specified C++ compiler name
        @return None if the command is not a compilation, otherwise
            (compiler_name, remaining_args)
            - compiler_name (str): the binary name of the compiler
              (e.g., 'clang++')
            - remaining_args (list(str)): all other arguments
              (e.g., ['-c', '-o', 'out.o'])
        '''

        if command:  # not empty list will allow to index '0' and '1:'
            executable = os.path.basename(command[0])  # type: str
            parameters = command[1:]  # type: List[str]
            # 'wrapper' 'parameters' and
            # 'wrapper' 'compiler' 'parameters' are valid.
            # Additionally, a wrapper can wrap another wrapper.
            if is_wrapper(executable):
                result = cls._split_compiler(parameters, cc, cxx)
                # Compiler wrapper without compiler is a 'C' compiler.
                return (executable, parameters) if result is None else result
            # MPI compiler wrappers add extra parameters
            elif is_mpi_wrapper(executable):
                # Pass the executable with full path to avoid pick different
                # executable from PATH.
                mpi_call = get_mpi_call(command[0])  # type: List[str]
                return cls._split_compiler(mpi_call + parameters, cc, cxx)
            # and 'compiler' 'parameters' is valid.
            elif is_c_compiler(executable, cc):
                return executable, parameters
            elif is_cxx_compiler(executable, cxx):
                return executable, parameters
        return None

def parse_args(arguments, prog=sys.argv[0]):
    'Parse arguments and return parsed args'
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''
            Captures source file compilations into a JSON database.  This can
            then be used by flit import and flit update to generate custom.mk.
            The JSON compilation database is similar to the format defined by
            Clang.
            ''',
        )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debugging information')
    parser.add_argument('-o', '--cdb', '--output', metavar='<file>',
                        dest='output', type=str,
                        default='compile_commands.json',
                        help='Output JSON compilation database file location.')
    parser.add_argument('--use-cc', metavar='<path>', dest='cc', type=str,
                        default=os.getenv('CC', 'cc'),
                        help='''
                            not yet implemented.
                            By default, will pull from the "CC" environment
                            variable if available.
                            ''')
    parser.add_argument('--use-c++', metavar='<path>', dest='cxx', type=str,
                        default=os.getenv('CXX', 'c++'),
                        help='''
                            not yet implemented.
                            By default, will pull from the "CXX" environment
                            variable if available.
                            ''')
    parser.add_argument('--append', action='store_true',
                        help='''
                            Extend existing compilation database with new
                            entries.  Duplicate entries are detected and not
                            present in the final output.  The output is not
                            continuously updated, it's done when the build
                            command finished.
                            ''')
    parser.add_argument(
        dest='build', nargs=argparse.REMAINDER,
        help='Command to run (e.g., make -j4)')
    args = parser.parse_args(arguments)
    if len(args.build) == 0:
        parser.error(message='missing build command')
    reconfigure_logging(int(args.verbose) * 2)
    return args

def is_wrapper(cmd):
    'Returns True if cmd is a known compiler wrapper'
    return COMPILER_PATTERN_WRAPPER.match(cmd) is not None

def is_mpi_wrapper(cmd):
    'Returns True if cmd is a known mpi wrapper'
    return COMPILER_PATTERNS_MPI_WRAPPER.match(cmd) is not None

def is_c_compiler(cmd, cc):
    'Returns True if cmd is a known C compiler'
    return os.path.basename(cc) == cmd or \
        any(pattern.match(cmd) is not None
            for pattern in COMPILER_PATTERNS_CC)

def is_cxx_compiler(cmd, cxx):
    'Returns True if cmd is a known C++ compiler'
    return os.path.basename(cxx) == cmd or \
        any(pattern.match(cmd) is not None
            for pattern in COMPILER_PATTERNS_CXX)

def capture(args):
    '''
    Runs the build and captures the calls to the compilers.

    @param args: the namespace object from parse_args()
    @return (exit_code, compilations)
    - exit_code: the exit code of the build
    - compilations: an iterable of libscanbuild.compilation.Compilation objects

    Copied largely from libscanbuild/intercept.py
    '''
    with temporary_directory(prefix='flit-capture-') as tmpdir:
        env = setup_environment(args, tmpdir)
        exit_code = run_build(args.build, env=env)
        calls = (parse_exec_trace(tracefile)
                 for tracefile in exec_trace_files(tmpdir))
        compilations = set(
            compilation for call in calls for compilation in
            CustomCompilation.iter_from_execution(call, args.cc, args.cxx)
            )

    return exit_code, iter(compilations)

def compilation2db_entry(compilation, cc, cxx):
    'Converts a compilation to a database entry for JSON'
    relative = os.path.relpath(compilation.source, compilation.directory)
    args = [compilation.compiler, '-c'] + compilation.flags + [relative]
    language = None
    if is_c_compiler(compilation.compiler, cc):
        language = 'c'
    elif is_cxx_compiler(compilation.compiler, cxx):
        language = 'c++'
    return {
        'compiler': compilation.compiler,
        'file': relative,
        'arguments': args,
        'directory': compilation.directory,
        'language': language,
        }

# TODO: handle other languages such as FORTRAN and CUDA
def main(arguments, prog=sys.argv[0]):
    '''
    Main logic here

    Copied largely from libscanbuild/intercept.py
    '''
    args = parse_args(arguments, prog)

    # Remove after testing
    args.override_compiler = False
    # Finish remove after testing

    exit_code, compilations = capture(args)

    to_save = compilations
    if args.append and os.path.isfile(args.output):
        prev_compilations = CompilationDatabase.load(args.output)
        to_save = iter(set(itertools.chain(prev_compilations, compilations)))

    entries = [compilation2db_entry(compilation, args.cc, args.cxx)
               for compilation in to_save]
    with open(args.output, 'w') as fout:
        json.dump(entries, fout, sort_keys=True, indent=4)

    return exit_code

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
