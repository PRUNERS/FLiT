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

'Implements the update subcommand'

import argparse
import os
import re
import subprocess as subp
import sys

import flitconfig as conf
import flitutil as util

brief_description = 'Updates the Makefile based on flit-config.toml'

def populate_parser(parser=None):
    'Populate or create an ArgumentParser'
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.description = '''
            Updates the Makefile based on flit-config.toml.  The Makefile
            is autogenerated and should not be modified manually.  If there
            are things you want to replace or add, you can use custom.mk
            which is included at the end of the Makefile.  So, you may add
            rules, add to variables, or override variables.
            '''
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    return parser

def flag_name(flag):
    '''
    Returns an associated Makefile variable name for the given compiler flag

    @param flag: (str) switches for the compiler

    @return (str) a valid Makefile variable unique to the given flag

    >>> flag_name('')
    'NO_FLAGS'

    >>> flag_name('-')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag only made of dashes

    >>> flag_name('----')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag only made of dashes

    >>> flag_name('-funsafe-math-optimizations')
    'FUNSAFE_MATH_OPTIMIZATIONS'

    >>> flag_name('-Ofast -march=32bit')
    'OFAST__MARCH_32BIT'

    >>> flag_name('-3')
    Traceback (most recent call last):
      ...
    AssertionError: Error: cannot handle flag that starts with a number

    >>> flag_name('-compiler-name=clang++')
    'COMPILER_NAME_CLANGpp'
    '''
    if flag == '':
        return 'NO_FLAGS'
    name = re.sub('[^0-9A-Za-z]', '_',
                  flag.upper().strip('-').replace('+', 'p'))
    assert re.match('^[0-9]', name) is None, \
        'Error: cannot handle flag that starts with a number'
    assert len(name) > 0, 'Error: cannot handle flag only made of dashes'
    return name

def gen_assignments(flag_map):
    '''
    Given a mapping of Makefile variable name to value, create a single string
    of assignments suitable for placing within a Makefile

    @note no checking is performed on the keys of the map.  They are assumed to
        be valid Makefile variables

    @param flag_map: ({str: str}) mapping from Makefile variable name to
        Makefile value.
    @return (str) The string to insert into a Makefile to create the
        assignments

    >>> gen_assignments({})
    ''

    >>> gen_assignments({'single_name': 'single_value'})
    'single_name     := single_value'

    Here we use an OrderedDict for the test to be robust.  If we used a normal
    dict, then the output lines could show up in a different order.
    >>> from collections import OrderedDict
    >>> print(gen_assignments(
    ...     OrderedDict([('hello', 'there'), ('my', 'friend')])))
    hello           := there
    my              := friend

    >>> print(gen_assignments(OrderedDict([
    ...     ('REALLY_A_VERY_LONG_VARIABLE_NAME_HERE', 'bob'),
    ...     ('not_so_long_32', 'harry'),
    ...     ('short', 'very long value here'),
    ...     ])))
    REALLY_A_VERY_LONG_VARIABLE_NAME_HERE := bob
    not_so_long_32  := harry
    short           := very long value here
    '''
    name_assignments = ['{} := {}'.format(name.ljust(15), flag)
                        for name, flag in flag_map.items()]
    return '\n'.join(name_assignments)

def gen_multi_assignment(name, values):
    '''
    Generates a multi-line assignment string for a Makefile

    @note no checking is done on the name or values to see if they are valid to
        place within a Makefile.

    @param name: (str) Makefile variable name
    @param values: (iter(str)) iterable of values to assign, one per line

    @return (str) a single string with the multi-line assignment suitable for a
        Makefile.

    >>> gen_multi_assignment('CLANG', None)
    'CLANG           :='

    >>> gen_multi_assignment('CLANG', [])
    'CLANG           :='

    >>> print(gen_multi_assignment('hello_there', ['my friend', 'my enemy']))
    hello_there     :=
    hello_there     += my friend
    hello_there     += my enemy
    '''
    values = values or tuple() # if None, set to an empty tuple
    justified = name.ljust(15)
    beginning = justified + ' :='
    return '\n'.join(
        [beginning] + ['{} += {}'.format(justified, x) for x in values])

def get_gcc_compiler_version(binary):
    'Return the version of the given gcc executable'
    return util.check_output([binary, '-dumpversion'])

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
        mpi_cxxflags = util.check_output(['mpic++', '--showme:compile'], *args, **kwargs)
        mpi_ldflags = util.check_output(['mpic++', '--showme:link'], *args, **kwargs)
    except subp.CalledProcessError:
        mpi_cxxflags = util.check_output(['mpic++', '-compile_info'], *args, **kwargs)
        mpi_ldflags = util.check_output(['mpic++', '-link_info'], *args, **kwargs)
        mpi_cxxflags = ' '.join(mpi_cxxflags.split()[2:])
        mpi_ldflags = ' '.join(mpi_ldflags.split()[2:])

    return mpi_cxxflags.split(), mpi_ldflags.split()

def _additional_ldflags(compiler):
    'Returns a list of LD flags needed for this particular compiler'
    if compiler['type'] == 'clang':
        return '-nopie'
    if compiler['type'] == 'intel':
        return '-no-pie'
    if compiler['type'] == 'gcc':
        version = get_gcc_compiler_version(compiler['binary'])
        major_version = version.split('.')[0]
        if int(major_version) >= 6:
            return '-no-pie'
        return ''
    raise NotImplementedError('Unsupported compiler type requested')

def create_makefile(args, makefile='Makefile'):
    'Create the makefile assuming flit-config.toml is in the current directory'
    projconf = util.load_projconf()
    compilers = {c['name']: c for c in projconf['compiler']}
    dev_build = projconf['dev_build']
    gt_build = projconf['ground_truth']
    try:
        dev_compiler = compilers[dev_build['compiler_name']]
        gt_compiler = compilers[gt_build['compiler_name']]
    except KeyError as ex:
        raise KeyError('Compiler name ' + ex.args[0] + ' not found')

    base_compilers = {x.upper(): None for x in util.SUPPORTED_COMPILER_TYPES}
    base_compilers.update({compiler['type'].upper(): compiler['binary']
                           for compiler in projconf['compiler']})

    test_run_args = ''
    if not projconf['run']['timing']:
        test_run_args = '--no-timing'
    else:
        test_run_args = ' '.join([
            '--timing-loops', str(projconf['run']['timing_loops']),
            '--timing-repeats', str(projconf['run']['timing_repeats']),
            ])

    mpi_cxxflags = []
    mpi_ldflags = []
    if projconf['run']['enable_mpi']:
        mpi_cxxflags, mpi_ldflags = get_mpi_flags()

    replacements = {
        'uname': os.uname().sysname,
        'hostname': os.uname().nodename,
        'dev_compiler': dev_compiler['binary'],
        'dev_type': dev_compiler['type'],
        'dev_optl': dev_build['optimization_level'],
        'dev_switches': dev_build['switches'],
        'dev_cxxflags': '$(' + dev_compiler['type'].upper() + '_CXXFLAGS)',
        'dev_ldflags': '$(' + dev_compiler['type'].upper() + '_LDFLAGS)',
        'ground_truth_compiler': gt_compiler['binary'],
        'ground_truth_type': gt_compiler['type'],
        'ground_truth_optl': gt_build['optimization_level'],
        'ground_truth_switches': gt_build['switches'],
        'gt_cxxflags': '$(' + gt_compiler['type'].upper() + '_CXXFLAGS)',
        'gt_ldflags': '$(' + gt_compiler['type'].upper() + '_LDFLAGS)',
        'flit_include_dir': conf.include_dir,
        'flit_data_dir': conf.data_dir,
        'flit_script_dir': conf.script_dir,
        'flit_version': conf.version,
        'flit_src_dir': conf.src_dir,
        'test_run_args': test_run_args,
        'enable_mpi': 'yes' if projconf['run']['enable_mpi'] else 'no',
        'mpi_cxxflags': ' '.join(mpi_cxxflags),
        'mpi_ldflags': ' '.join(mpi_ldflags),
        'compiler_defs': gen_assignments({
            key: val for key, val in base_compilers.items()}),
        'compilers': ' '.join([compiler['type'].upper()
                               for compiler in projconf['compiler']]),
        'opcodes_definitions': gen_assignments({
            flag_name(x): x
            for compiler in projconf['compiler']
            for x in compiler['optimization_levels']}),
        'switches_definitions': gen_assignments({
            flag_name(x): x
            for compiler in projconf['compiler']
            for x in compiler['switches_list']}),
        'compiler_opcodes': '\n\n'.join([
            gen_multi_assignment(
                'OPCODES_' + compiler['type'].upper(),
                [flag_name(x) for x in compiler['optimization_levels']])
            for compiler in projconf['compiler']]),
        'compiler_switches': '\n\n'.join([
            gen_multi_assignment(
                'SWITCHES_' + compiler['type'].upper(),
                [flag_name(x) for x in compiler['switches_list']])
            for compiler in projconf['compiler']]),
        'compiler_fixed_compile_flags': gen_assignments({
            compiler['type'].upper() + '_CXXFLAGS':
                compiler['fixed_compile_flags']
            for compiler in projconf['compiler']}),
        'compiler_fixed_link_flags': gen_assignments({
            compiler['type'].upper() + '_LDFLAGS':
                compiler['fixed_link_flags'] + ' '
                + _additional_ldflags(compiler)
            for compiler in projconf['compiler']}),
        }

    util.process_in_file(os.path.join(conf.data_dir, 'Makefile.in'),
                         makefile, replacements, overwrite=True)

def main(arguments, prog=None):
    'Main logic here'
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    makefile = os.path.join(args.directory, 'Makefile')
    if os.path.exists(makefile):
        print('Updating {0}'.format(makefile))
    else:
        print('Creating {0}'.format(makefile))

    with util.pushd(args.directory):
        try:
            create_makefile(args)
        except FileNotFoundError as ex:
            print('Error {}:'.format(ex.errno), ex.strerror,
                  '"{}"'.format(ex.filename),
                  file=sys.stderr)
            return 1
        except (KeyError, AssertionError) as ex:
            print('Error:', ex.args[0], file=sys.stderr)
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
