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

from ninja_syntax import as_list, Writer

import argparse
import os
import re
import subprocess as subp
import sys
from socket import gethostname

import flitutil as util
import flitconfig as conf

brief_description = 'Generate Ninja build file for FLiT makefile system'

BUILD_FILENAME = 'build.ninja'

def parse_args(arguments, prog=sys.argv[0]):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(
        prog=prog,
        description='''
            Generates a Ninja build file instead of a GNU Makefile for
            performing the FLiT build in a FLiT test directory.
            ''',
        )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to genreate build.ninja')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args(arguments)
    return args

def check_output(*args, **kwargs):
    output = subp.check_output(*args, **kwargs)
    return output.decode(encoding='utf-8')

def variablize(name):
    '''
    Convert the name to a valid variable name

    >>> variablize('')
    'NO_FLAGS'

    >>> variablize('-')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag only made of dashes

    >>> variablize('----')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag only made of dashes

    >>> variablize('-funsafe-math-optimizations')
    'FUNSAFE_MATH_OPTIMIZATIONS'

    >>> variablize('-Ofast -march=32bit')
    'OFAST__MARCH_32BIT'

    >>> variablize('-3')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag that starts with a number

    >>> variablize('-compiler-name=clang++')
    'COMPILER_NAME_CLANGpp'
    '''
    if name == '':
        return 'NO_FLAGS'
    name = re.sub('[^0-9A-Za-z]', '_',
                  name.upper().strip('-').replace('+', 'p'))
    assert re.match('^[0-9]', name) is None, \
        'Error: cannot handle name that starts with a number'
    assert len(name) > 0, 'Error: cannot handle name only made of dashes'
    return name

def get_mpi_flags():
    '''
    Returns both cxxflags and ldflags for mpi compilation

    @return (cxxflags, ldflags)
      cxxflags (list(str)) list of flags for the c++ compiler for MPI
      ldflags (list(str)) list of flags for the linker for MPI
    '''
    try:
        mpi_cxxflags = check_output(['mpic++', '--showme:compile'])
        mpi_ldflags = check_output(['mpic++', '--showme:link'])
    except subp.CalledProcessError:
        mpi_cxxflags = check_output(['mpic++', '-compile_info'])
        mpi_ldflags = check_output(['mpic++', '-link_info'])
        mpi_cxxflags = ' '.join(mpi_cxxflags.split()[2:])
        mpi_ldflags = ' '.join(mpi_ldflags.split()[2:])

    return mpi_cxxflags.split(), mpi_ldflags.split()

def get_gcc_compiler_version(binary):
    return check_output([binary, '-dumpversion'])

class NinjaWriter:
    def __init__(self, out, prog=sys.argv[0], arguments=sys.argv[1:]):
        self.writer = Writer(out)
        self.prog = prog
        self.ninja_required_version = '1.3'
        self.ninja_gen_deps = []
        self.configure_args = arguments
        self.hostname = gethostname()
        self.sources = []
        self.cxxflags = [
            '-fno-pie',
            '-std=c++11',
            '-I.',
            '-I' + conf.include_dir,
            ]
        self.ldflags = [
            '-lm',
            '-lstdc++',
            '-L' + conf.lib_dir,
            '-Wl,-rpath=' + os.path.abspath(conf.lib_dir),
            '-lflit',
            ]
        self.compilers = []
        self.gt_compiler = None

    def load_makefile(self, makefile):
        self.ninja_gen_deps.append(makefile)
        makevars = util.extract_make_vars(makefile)
        self.sources.extend(sorted(makevars['SOURCE']))
        self.cxxflags.extend(makevars['CXXFLAGS'])
        self.ldflags.extend(makevars['LDFLAGS'])
        self.ldflags.extend(makevars['LDLIBS'])

    def load_project_config(self, tomlfile):
        self.ninja_gen_deps.append(tomlfile)
        projconf = util.load_projconf()

        if projconf['run']['enable_mpi']:
            mpi_cxxflags, mpi_ldflags = get_mpi_flags()
            self.cxxflags.extend(mpi_cxxflags)
            self.ldflags.extend(mpi_ldflags)

        # different compilers link differently for no position independence
        self.compilers = [dict(x) for x in projconf['compiler']]
        for compiler in self.compilers:
            if compiler['type'] == 'clang':
                compiler['fixed_link_flags'] += ' -nopie'
            if compiler['type'] == 'gcc':
                version = get_gcc_compiler_version(compiler['binary'])
                if version.split('.')[0] not in ('4', '5'):
                    compiler['fixed_link_flags'] += ' -no-pie'

        self.gt_compiler = dict(projconf['ground_truth'])
        self.gt_compiler['optl'] = self.gt_compiler['optimization_level']

        matching_gt_compiler = [
            x for x in self.compilers
            if x['name'] == self.gt_compiler['compiler_name']
            ]
        assert len(matching_gt_compiler) == 1
        matching_gt_compiler = matching_gt_compiler[0]

        self.gt_compiler['binary'] = matching_gt_compiler['binary']
        self.gt_compiler['cxxflags'] = \
            matching_gt_compiler['fixed_compile_flags']
        self.gt_compiler['ldflags'] = matching_gt_compiler['fixed_link_flags']

        # TODO: self.gt_compiler, with:
        # TODO-   - 'binary'
        # TODO-   - 'optl'
        # TODO-   - 'switches'
        # TODO-   - 'cxxflags' (containing compiler's cxxflags)
        # TODO-   - 'ldflags'  (containing compiler's ldflags)

    def _cxx_command(self, outdir, cxx, optl, switches, cxxflags, target):
        '''
        Generates a list of pieces that constitutes a compile command for Ninja
        to make a single object file

        @param outdir: output directory of object files (e.g., 'obj/gt')
        @param cxx: compiler binary (e.g., 'g++')
        @param optl: optimization level (e.g., '-O2')
        @param switches: switches under test (e.g., '-ffast-math')
        @param cxxflags: other flags not under test (e.g., '-fno-pie')
        @param target: name of the final executable this object file will be a
            part of, without the directory portion (e.g., 'devrun')
        '''
        command = [
            'mkdir -p', outdir, '&&',
            cxx, '-c $in -o $out',
            '-MMD -MT $out -MF $out.d',
            optl,
            ]
        command.extend(as_list(switches))
        command.extend(as_list(cxxflags))
        command.extend([
            '$cxxflags',
            '-DFLIT_HOST=\'"{}"\''.format(self.hostname),
            '-DFLIT_COMPILER=\'"{}"\''.format(cxx),
            '-DFLIT_OPTL=\'"{}"\''.format(optl),
            '-DFLIT_SWITCHES=\'"{}"\''.format(switches),
            '-DFLIT_FILENAME=\'"{}"\''.format(target),
            ])
        return command

    def _link_command(self, cxx, ldflags, outdir=None):
        'Generate the link command for Ninja files'
        command = []
        if outdir:
            command.append('mkdir -p {} && ')
        command.extend([cxx, '-o $out $in'])
        command.extend(as_list(ldflags))
        command.append('$ldflags')
        return command
            
    def write(self):
        'creates the ninja build file'
        n = self.writer

        n.comment('Autogenerated by Michael Bentley\'s script')
        n.comment('This file is to build the mfem tests')
        n.newline()

        n.variable('ninja_required_version', self.ninja_required_version)
        n.newline()

        n.comment('Arguments passed to the configure script:')
        n.variable('configure_args', self.configure_args)
        n.newline()

        n.variable('hostname', self.hostname)
        n.newline()

        n.variable('cxxflags', self.cxxflags)
        n.variable('ldflags', self.ldflags)
        n.newline()

        n.comment('Be able to reconfigure myself if needed')
        n.rule('configure_ninja',
               command=[self.prog, '$configure_args -q'],
               description='UPDATING $out',
               generator=True)
        n.newline()

        n.build('build.ninja', 'configure_ninja', implicit=self.ninja_gen_deps)
        n.newline()

        n.comment('Target to clean up')
        n.rule('CLEAN',
               command=['ninja', '-t', 'clean', '&&', 'rm', '-rf', '$toclean'],
               description='CLEANING UP')
        n.newline()

        n.build('clean', 'CLEAN', variables={'toclean': ['obj']})
        n.build('distclean', 'CLEAN',
                variables={'toclean': [
                    'obj', 'results', 'bin', 'devrun', 'gtrun',
                    'ground-truth.csv', 'ground-truth.csv*.dat']})

        #for compiler in self.compilers:
        #    name = variablize(compiler['name'])
        #    n.variable(name, compiler['binary'])
        #    n.variable(name + '_cxxflags', compiler['fixed_compile_flags'])
        #    n.variable(name + '_ldflags', compiler['fixed_link_flags'])
        #    n.newline()

        if self.gt_compiler is not None:
            n.rule('gt_cxx',
                   command=self._cxx_command(
                       outdir='obj/gt',
                       cxx=self.gt_compiler['binary'],
                       optl=self.gt_compiler['optl'],
                       switches=self.gt_compiler['switches'],
                       cxxflags=self.gt_compiler['cxxflags'],
                       target='gtrun'),
                   description='CXX $out',
                   depfile='$out.d',
                   deps='gcc')
            n.newline()

            n.rule('gt_link',
                   command=self._link_command(
                       cxx=self.gt_compiler['binary'],
                       ldflags=self.gt_compiler['ldflags']),
                   description='LINK $out')
            n.newline()

            n.build('gt', 'phony', 'gtrun')
            n.build('gtrun', 'gt_link',
                    inputs=['obj/gt/' + os.path.basename(x) + '.o'
                            for x in self.sources])
            n.newline()

            for source in self.sources:
                n.build('obj/gt/' + os.path.basename(source) + '.o',
                        'gt_cxx', source)

        # TODO: add ground-truth.csv
        # TODO: add dev and devrun
        # TODO: add runbuild and run
        # TODO: add rule to update build.ninja


def main(arguments, prog=sys.argv[0]):
    args = parse_args(arguments, prog=prog)
    arguments = [x for x in arguments if x not in ('-q', '--quiet')]

    if not args.quiet:
        if os.path.exists(BUILD_FILENAME):
            print('Updating', BUILD_FILENAME)
        else:
            print('Creating', BUILD_FILENAME)

    with open(BUILD_FILENAME, 'w') as build_file:
        writer = NinjaWriter(build_file, prog, arguments)
        writer.load_project_config('flit-config.toml')
        writer.load_makefile('custom.mk')
        writer.write()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
