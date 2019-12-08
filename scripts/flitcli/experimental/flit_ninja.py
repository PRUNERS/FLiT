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

try:
    import flitutil as util
    import flitconfig as conf
except ModuleNotFoundError:
    sys.path.append('..')
    import flitutil as util
    import flitconfig as conf

brief_description = 'Generate Ninja build file for FLiT makefile system'

BUILD_FILENAME = 'build.ninja'

def parse_args(arguments, prog=sys.argv[0]):
    '''
    Parse command-line arguments

    >>> parse_args([])
    Namespace(directory='.', quiet=False)

    >>> parse_args(['-C', 'my/dir', '-q'])
    Namespace(directory='my/dir', quiet=True)

    >>> parse_args(['--directory', 'another/dir', '--quiet'])
    Namespace(directory='another/dir', quiet=True)
    '''
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
    '''
    Wrapper around subprocess.check_output() that returns a str object

    >>> check_output(['echo', 'hello there'])
    'hello there\\n'
    '''
    output = subp.check_output(*args, **kwargs)
    return output.decode(encoding='utf-8')

def variablize(name):
    '''
    Convert the name to a valid variable name

    >>> variablize('')
    'NO_FLAGS'

    >>> variablize('-')
    '_'

    >>> variablize('----')
    '____'

    >>> variablize('-funsafe-math-optimizations')
    '_FUNSAFE_MATH_OPTIMIZATIONS'

    >>> variablize('-Ofast -march=32bit')
    '_OFAST__MARCH_32BIT'

    >>> variablize('-3')
    '_3'

    >>> variablize('-compiler-name=clang++')
    '_COMPILER_NAME_CLANGxx'
    '''
    if name == '':
        return 'NO_FLAGS'
    name = re.sub('[^0-9A-Za-z]', '_',
                  name.upper().replace('+', 'x'))
    assert re.match('^[0-9]', name) is None, \
        'Error: cannot handle name that starts with a number'
    assert len(name) > 0, 'Error: cannot handle name only made of dashes'
    return name

def get_mpi_flags(*args, **kwargs):
    '''
    Returns both cxxflags and ldflags for mpi compilation

    @param args, kwargs: extra arguments passed to subprocess.Popen()
    @return (cxxflags, ldflags)
      cxxflags (list(str)) list of flags for the c++ compiler for MPI
      ldflags (list(str)) list of flags for the linker for MPI

    Setup a fake mpic++ executable
    >>> import tempfile
    >>> import shutil
    >>> tempdir = tempfile.mkdtemp()
    >>> with open(os.path.join(tempdir, 'mpic++'), 'w') as fout:
    ...     _ = fout.write('#!{}\\n'.format(shutil.which('python3')))
    ...     _ = fout.write('import sys\\n')
    ...     _ = fout.write('if "compile" in sys.argv[1]: print("compile arguments")\\n')
    ...     _ = fout.write('if "link" in sys.argv[1]: print("link arguments")\\n')
    >>> os.chmod(os.path.join(tempdir, 'mpic++'), mode=0o755)

    Test that compile and link flags are returned
    >>> cxxflags, ldflags = get_mpi_flags(env={'PATH': tempdir})
    >>> cxxflags
    ['compile', 'arguments']
    >>> ldflags
    ['link', 'arguments']

    Cleanup
    >>> shutil.rmtree(tempdir)
    '''
    try:
        mpi_cxxflags = check_output(['mpic++', '--showme:compile'], *args, **kwargs)
        mpi_ldflags = check_output(['mpic++', '--showme:link'], *args, **kwargs)
    except subp.CalledProcessError:
        mpi_cxxflags = check_output(['mpic++', '-compile_info'], *args, **kwargs)
        mpi_ldflags = check_output(['mpic++', '-link_info'], *args, **kwargs)
        mpi_cxxflags = ' '.join(mpi_cxxflags.split()[2:])
        mpi_ldflags = ' '.join(mpi_ldflags.split()[2:])

    return mpi_cxxflags.split(), mpi_ldflags.split()

def get_gcc_compiler_version(binary):
    'Return the version of the given gcc executable'
    return check_output([binary, '-dumpversion'])

def _create_compilation(compiler, optl, switches):
    '''
    Create compilation dictionary for the given compilation

    A compilation has:

    - id
    - compiler_name
    - binary
    - optl
    - switches
    - cxxflags
    - ldflags
    - target
    - resultsfile

    @param compiler (dict(str->str)): a dictionary with keys
        - name
        - binary
        - fixed_compile_flags
        - fixed_link_flags
    @param optl (str): optimization level
    @param switches (str or list(str)): flags under test

    >>> compiler = {'name': 'N', 'binary': './N',
    ...             'fixed_compile_flags': 'fc', 'fixed_link_flags': 'fl'}
    >>> c = _create_compilation(compiler, '-O3', '-ffast-math -mavx2')
    >>> c['id']
    'N_O3_FFAST_MATH__MAVX2'
    >>> c['compiler_name']
    'N'
    >>> c['binary']
    './N'
    >>> c['optl']
    '-O3'
    >>> c['switches']
    '-ffast-math -mavx2'
    >>> c['cxxflags']
    'fc'
    >>> c['ldflags']
    'fl'
    >>> c['target']
    'bin/N_O3_FFAST_MATH__MAVX2'
    >>> c['resultsfile']
    'results/N_O3_FFAST_MATH__MAVX2-out'
    '''
    v_compiler_name = variablize(compiler['name'])
    v_optl = variablize(optl)
    v_switches = variablize(switches)
    my_id = v_compiler_name + v_optl + v_switches

    compilation = {
        'id': my_id,
        'compiler_name': compiler['name'],
        'binary': compiler['binary'],
        'optl': optl,
        'switches': switches,
        'cxxflags': compiler['fixed_compile_flags'],
        'ldflags': compiler['fixed_link_flags'],
        'target': os.path.join('bin', my_id),
        'resultsfile': os.path.join('results', my_id + '-out'),
        }

    return compilation

class NinjaWriter:
    '''
    Output to a Ninja build file.

    The following attributes are available:

    - writer: internal output interface implementing:
      - comment()
      - variable()
      - rule()
      - build()
      - newline()
    - prog: executable to run this configure script
    - ninja_required_version: ninja version that is required
    - ninja_gen_deps: dependencies for the build.ninja file
    - configure_args: arguments passed to this script
    - hostname: hostname of the system
    - sources: list of source files
    - cxxflags: c++ compile flags
    - ldflags: c++ link flags
    - compilers: list of compilers with settings
    - gt_compilation: compilation settings for the baseline compilation
    - run_wrapper: executable to wrap test executables when running
    - timing_flags: flags for timing for test executables
    '''

    def __init__(self, out, prog=sys.argv[0], arguments=sys.argv[1:]):
        '''
        Initialize Ninja Writer

        @param out: output file object with write() function
        @param prog: executable responsible for calling this script
        @param arguments: arguments passed to this script
        '''
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
        self.compilers = {}
        self.gt_compilation = None
        self.run_wrapper = ''
        self.timing_flags = ''
        self._written_rules = set()

    def load_makefile(self, makefile):
        '''
        Load Makefile and extract variables.  The variables pulled out are:
        - SOURCE: list of c++ source files
        - CXXFLAGS: c++ compiler flags
        - LDFLAGS: c++ linker flags
        - LDLIBS: c++ libraries to link (e.g., '-lm')
        - RUN_WRAPPER: executable to wrap the running of the test executables

        Test of an empty file
        >>> from tempfile import NamedTemporaryFile
        >>> import io
        >>> w = NinjaWriter(io.StringIO())

        >>> cxxflags_orig = list(w.cxxflags)
        >>> ldflags_orig = list(w.ldflags)
        
        >>> with NamedTemporaryFile() as makefile_out:
        ...     w.load_makefile(makefile_out.name)

        >>> w.sources
        []
        >>> w.cxxflags == cxxflags_orig
        True
        >>> w.ldflags == ldflags_orig
        True
        >>> w.run_wrapper
        ''

        Test of a simple file
        >>> from tempfile import NamedTemporaryFile
        >>> with io.StringIO() as writer_out:
        ...     w = NinjaWriter(writer_out)

        >>> cxxflags_orig = list(w.cxxflags)
        >>> ldflags_orig = list(w.ldflags)

        >>> with NamedTemporaryFile(mode='w') as makefile_out:
        ...     _ = makefile_out.write('SOURCE   := a.cpp\\n')
        ...     _ = makefile_out.write('SOURCE   += b.cpp\\n')
        ...     _ = makefile_out.write('CXXFLAGS  = -std=c++11 -Werror\\n')
        ...     _ = makefile_out.write('LDFLAGS  += -L/usr/local/lib64 -L/opt/gcc\\n')
        ...     _ = makefile_out.write('LDLIBS    = -lm\\n')
        ...     _ = makefile_out.write('LDLIBS   += -lflit\\n')
        ...     _ = makefile_out.write('RUN_WRAPPER := /usr/bin/echo -ne \\n')
        ...     makefile_out.flush()
        ...     w.load_makefile(makefile_out.name)

        >>> w.sources
        ['a.cpp', 'b.cpp']
        >>> w.cxxflags == cxxflags_orig + ['-std=c++11', '-Werror']
        True
        >>> w.ldflags == ldflags_orig + [
        ...     '-L/usr/local/lib64', '-L/opt/gcc', '-lm', '-lflit']
        True
        >>> w.run_wrapper
        '/usr/bin/echo -ne'
        '''
        self.ninja_gen_deps.append(makefile)
        makevars = util.extract_make_vars(makefile)
        if 'SOURCE' in makevars: self.sources.extend(sorted(makevars['SOURCE']))
        if 'CXXFLAGS' in makevars: self.cxxflags.extend(makevars['CXXFLAGS'])
        if 'LDFLAGS' in makevars: self.ldflags.extend(makevars['LDFLAGS'])
        if 'LDLIBS' in makevars: self.ldflags.extend(makevars['LDLIBS'])
        if 'RUN_WRAPPER' in makevars:
            self.run_wrapper = ' '.join(makevars['RUN_WRAPPER'])

    def load_project_config(self, tomlfile):
        'Load configuration from flit-config.toml'
        # TODO: write tests
        self.ninja_gen_deps.append(tomlfile)
        projconf = util.load_projconf()

        self.hostname = projconf['host']['name']

        if projconf['run']['enable_mpi']:
            mpi_cxxflags, mpi_ldflags = get_mpi_flags()
            self.cxxflags.extend(mpi_cxxflags)
            self.ldflags.extend(mpi_ldflags)

        # different compilers link differently for no position independence
        self.compilers = {x['name']: dict(x) for x in projconf['compiler']}
        for compiler in self.compilers.values():
            if compiler['type'] == 'clang':
                compiler['fixed_link_flags'] += ' -nopie'
            if compiler['type'] == 'gcc':
                version = get_gcc_compiler_version(compiler['binary'])
                if version.split('.')[0] not in ('4', '5'):
                    compiler['fixed_link_flags'] += ' -no-pie'

        self.gt_compilation = _create_compilation(
            self.compilers[projconf['ground_truth']['compiler_name']],
            projconf['ground_truth']['optimization_level'],
            projconf['ground_truth']['switches'])
        self.gt_compilation['id'] = 'gt'
        self.gt_compilation['target'] = 'gtrun'
        self.gt_compilation['resultsfile'] = 'ground-truth.csv'

        self.dev_compilation = _create_compilation(
            self.compilers[projconf['dev_build']['compiler_name']],
            projconf['dev_build']['optimization_level'],
            projconf['dev_build']['switches'])
        self.dev_compilation['id'] = 'dev'
        self.dev_compilation['target'] = 'devrun'
        self.dev_compilation['resultsfile'] = 'devrun.csv'

        if not projconf['run']['timing']:
            self.timing_flags = '--no-timing'
        else:
            self.timing_flags = '--timing-repeats {} --timing-loops {}'.format(
                projconf['run']['timing_repeats'],
                projconf['run']['timing_loops'])


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
        # TODO: write tests
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
        # TODO: write tests
        command = []
        if outdir:
            command.append('mkdir -p {} && ')
        command.extend([cxx, '-o $out $in'])
        command.extend(as_list(ldflags))
        command.append('$ldflags')
        return command

    def _write_help(self):
        'Writes the help target to the ninja build file'
        # TODO: write tests
        self.writer.comment('Print help to the user')
        self.writer.rule(
            'HELP',
            command=[
                'echo', '&&',
                'echo', '"The following targets are available."', '&&',
                'echo', '&&',
                'echo', '" - help ....... Show this help and exit (default target)"',
                '&&',
                'echo', '" - dev ........ Only run the devel compilation to test things out"',
                '&&',
                'echo', '" - gt ......... Compile the gtrun executable"',
                '&&',
                'echo', '" - runbuild ... Build all executables needed for the run target"',
                '&&',
                'echo', '" - run ........ Run all combinations of compilation, results in results/"',
                '&&',
                'echo', '" - clean ...... Clean intermediate files"',
                '&&',
                'echo', '" - veryclean .. Runs clean + removes targets and results"',
                '&&',
                'echo', '" - distclean .. Same as veryclean"',
                '&&',
                'echo',
                ],
            description='DISPLAY help')
        self.writer.build('help', 'HELP')
        self.writer.newline()

    def _write_clean(self):
        'Writes the clean targets to the ninja build file'
        # TODO: write tests
        self.writer.comment('Target to clean up')
        self.writer.rule('CLEAN',
               command=['ninja', '-t', 'clean', '&&', 'rm', '-rf', '$toclean'],
               description='CLEANING UP')
        self.writer.newline()

        self.writer.build('clean', 'CLEAN', variables={'toclean': ['obj']})
        self.writer.build('veryclean', 'CLEAN',
                variables={'toclean': [
                    'obj', 'results', 'bin', 'devrun', 'gtrun',
                    'ground-truth.csv', 'ground-truth.csv*.dat']})
        self.writer.build('distclean', 'phony', 'veryclean')
        self.writer.newline()

    def _try_rule(self, name, *args, **kwargs):
        '''
        Write rule to Ninja file only if _try_rule() was not called for this
        rule already
        '''
        # TODO: write tests
        if name not in self._written_rules:
            self._written_rules.add(name)
            self.writer.rule(name, *args, **kwargs)

    def _write_compilation(self, compilation):
        '''
        Writes the compilation to the ninja build file

        The compilation is a dictionary with the following keys:

        - id: unique identifier for this compilation
        - compiler_name: name of the compiler used
        - binary: name or path of the compiler executable
        - optl: optimization level
        - switches: switches under test
        - cxxflags: other compiler flags (including compiler-specific)
        - target: file path of destination executable
        - ldflags: link flags (including compiler-specific)
        - resultsfile: file path to store results of running
        '''
        # TODO: write tests
        n = self.writer

        name = compilation['id']
        compile_rule_name = name + '_cxx'
        link_rule_name = variablize(compilation['compiler_name']) + '_link'
        obj_dir = os.path.join('obj', name)

        # TODO: implement a pool for run_tests since it is timing-sensative??
        self._try_rule('run_tests',
                       command=[
                           'mkdir -p results &&',
                           self.run_wrapper,
                           os.path.join('.', '$in'),
                           '$timing_flags',
                           '-o $out',
                           ],
                       description='TEST OUTPUT $out')
        n.newline()

        self._try_rule(compile_rule_name,
                       command=self._cxx_command(
                           outdir=obj_dir,
                           cxx=compilation['binary'],
                           optl=compilation['optl'],
                           switches=compilation['switches'],
                           cxxflags=compilation['cxxflags'],
                           target=os.path.basename(compilation['target'])),
                       description='CXX $out',
                       depfile='$out.d',
                       deps='gcc')
        n.newline()

        self._try_rule(link_rule_name,
                       command=self._link_command(
                           cxx=compilation['binary'],
                           ldflags=compilation['ldflags']),
                       description='LINK $out')
        n.newline()

        n.build(compilation['target'], link_rule_name,
                inputs=[os.path.join(obj_dir, os.path.basename(x) + '.o')
                        for x in self.sources])
        n.newline()

        for source in self.sources:
            n.build(os.path.join(obj_dir, os.path.basename(source) + '.o'),
                    compile_rule_name, source)
        n.newline()

        n.build(compilation['resultsfile'], 'run_tests', compilation['target'])


    def write(self):
        'creates the ninja build file'
        # TODO: write tests
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

        n.comment('Timing flags for running the tests')
        n.variable('timing_flags', self.timing_flags)
        n.newline()

        n.comment('Be able to reconfigure myself if needed')
        n.rule('configure_ninja',
               command=[self.prog, '$configure_args -q'],
               description='UPDATING $out',
               generator=True)
        n.newline()

        n.build('build.ninja', 'configure_ninja', implicit=self.ninja_gen_deps)
        n.newline()

        self._write_help()
        n.default('help')
        n.newline()

        self._write_clean()

        comparison_suffix = '-comparison.csv'
        n.rule('compare',
               command=[
                   self.run_wrapper,
                   os.path.join('.', self.gt_compilation['target']),
                   '--compare-mode',
                   '--compare-gt', self.gt_compilation['resultsfile'],
                   '--suffix "{}"'.format(comparison_suffix),
                   '$in',
                   '-o /dev/null',
                   ],
               description='COMPARE TO $out')
        n.newline()

        if self.gt_compilation is not None:
            self._write_compilation(self.gt_compilation)
            n.build('gt', 'phony', 'gtrun')
            n.newline()

        if self.dev_compilation is not None:
            self._write_compilation(self.dev_compilation)
            n.build('dev', 'phony', 'devrun')
            n.newline()

        runbuild_targets = ['gt']
        results_files = []
        for name, compiler in self.compilers.items():
            for optl in compiler['optimization_levels']:
                for switches in compiler['switches_list']:
                    compilation = _create_compilation(
                        compiler, optl, switches)
                    runbuild_targets.append(compilation['target'])
                    results_files.append(compilation['resultsfile'])
                    self._write_compilation(compilation)
                    n.newline()
        n.build('runbuild', 'phony', runbuild_targets)
        n.newline()

        for results_file in results_files:
            n.build(results_file + comparison_suffix, 'compare', results_file,
                    implicit=[
                        self.gt_compilation['resultsfile'],
                        self.gt_compilation['target'],
                        ])
        n.newline()
        n.build('run', 'phony', [x + comparison_suffix for x in results_files])

def main(arguments, prog=sys.argv[0]):
    'Main logic here'
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
        if os.path.isfile('custom.mk'):
            writer.load_makefile('custom.mk')
        writer.write()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
