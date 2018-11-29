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

'Implements the update subcommand'

import argparse
import os
import re
import sys
import toml

import flitconfig as conf
import flitutil

brief_description = 'Updates the Makefile based on flit-config.toml'
_supported_compiler_types = ('clang', 'gcc', 'intel')

def parse_args(arguments, prog=sys.argv[0]):
    'Return parsed arugments'
    parser = argparse.ArgumentParser(
        prog=prog,
        description='''
            Updates the Makefile based on flit-config.toml.  The Makefile
            is autogenerated and should not be modified manually.  If there
            are things you want to replace or add, you can use custom.mk
            which is included at the end of the Makefile.  So, you may add
            rules, add to variables, or override variables.
            ''',
        )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    args = parser.parse_args(arguments)
    return args

def load_projconf(directory):
    '''
    Loads and returns the project configuration found in the given tomlfile.
    This function checks for validity of that tomlfile and fills it with
    default values.

    @param directory: directory containing 'flit-config.toml'.

    @return project configuration as a struct of dicts and lists depending on
    the structure of the given tomlfile.
    '''
    tomlfile = os.path.join(directory, 'flit-config.toml')
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        raise

    defaults = flitutil.get_default_toml()

    if 'compiler' in projconf:
        assert isinstance(projconf['compiler'], list), \
            'flit-config.toml improperly configured, ' \
            'needs [[compiler]] section'

        default_type_map = {c['type']: c for c in defaults['compiler']}
        type_map = {} # type -> compiler
        name_map = {} # name -> compiler
        for compiler in projconf['compiler']:

            # make sure each compiler has a name, type, and binary
            for field in ('name', 'type', 'binary'):
                assert field in compiler, \
                    'flit-config.toml: compiler "{0}"'.format(compiler) + \
                    ' is missing the "{0}" field'.format(field)

            # check that the type is valid
            assert compiler['type'] in _supported_compiler_types, \
                'flit-config.toml: unsupported compiler type "{0}"' \
                .format(compiler['type'])

            # check that we only have one of each type specified
            assert compiler['type'] not in type_map, \
                'flit-config.toml: cannot have multiple compilers of the ' \
                'same type ({0})'.format(compiler['type'])
            type_map[compiler['type']] = compiler

            # check that we only have one of each name specified
            assert compiler['name'] not in name_map, \
                'flit-config.toml: cannot have multiple compilers of the ' \
                'same name ({0})'.format(compiler['name'])
            name_map[compiler['name']] = compiler

            # if optimization_levels or switches_list are missing for any
            # compiler, put in the default flags for that compiler
            default = default_type_map[compiler['type']]
            for field in ('optimization_levels', 'switches_list'):
                if field not in compiler:
                    compiler[field] = default[field]

    # Fill in the rest of the default values
    flitutil.fill_defaults(projconf, defaults)

    return projconf

def flag_name(flag):
    if flag == '':
        return 'NO_FLAGS'
    name = re.sub('[^0-9A-Za-z]', '_', flag.upper().strip('-'))
    assert re.match('^[0-9]', name) is None, \
        'Error: cannot handle flag that starts with a number'
    assert len(name) > 0, 'Error: cannot handle flag only made of dashes'
    return name

def generate_assignments(flags):
    name_assignments = [name + ' := ' + flags[name] 
                        for name in flags.keys() if name != '']
    return '\n'.join(name_assignments)

def gen_optl_switches(compiler):
    '''
    From the compiler specification, generate Makefile strings for optimization
    levels and switches.

    @param compiler {'type': str,
                     'optimization_levels': list,
                     'switches_list': list}

    @return string to insert into the Makefile

    >>> gen_optl_switches({
    ...     'type': 'clang',
    ...     'optimization_levels': ['-O2', '-O3'],
    ...     'switches_list': ['-hi there', '-hello', '']
    ...     })
    CLANG_OPTL01    := -O2
    CLANG_OPTL02    := -O3
    CLANG_SWITCH01  := -hi there
    CLANG_SWITCH02  := -hello
    CLANG_SWITCH03  := 
    OPCODES_CLANG   :=
    OPCODES_CLANG   += CLANG_OPTL01
    OPCODES_CLANG   += CLANG_OPTL02
    SWITCHES_CLANG  :=
    SWITCHES_CLANG  += CLANG_SWITCH01
    SWITCHES_CLANG  += CLANG_SWITCH02
    SWITCHES_CLANG  += CLANG_SWITCH03
    '''
    pass

def main(arguments, prog=sys.argv[0]):
    'Main logic here'
    args = parse_args(arguments, prog=prog)

    try:
        projconf = load_projconf(args.directory)
    except FileNotFoundError:
        return 1
    except AssertionError as ex:
        print('Error: ' + ex.args[0], file=sys.stderr)
        return 1

    makefile = os.path.join(args.directory, 'Makefile')
    if os.path.exists(makefile):
        print('Updating {0}'.format(makefile))
    else:
        print('Creating {0}'.format(makefile))

    dev_build = projconf['dev_build']
    matching_dev_compilers = [x for x in projconf['compiler']
                              if x['name'] == dev_build['compiler_name']]
    assert len(matching_dev_compilers) > 0, \
            'Compiler name {0} not found'.format(dev_build['compiler_name'])
    assert len(matching_dev_compilers) < 2, \
            'Multiple compilers with name {0} found' \
            .format(dev_build['compiler_name'])

    ground_truth = projconf['ground_truth']
    matching_gt_compilers = [x for x in projconf['compiler']
                             if x['name'] == ground_truth['compiler_name']]
    assert len(matching_gt_compilers) > 0, \
            'Compiler name {0} not found'.format(ground_truth['compiler_name'])
    assert len(matching_gt_compilers) < 2, \
            'Multiple compilers with name {0} found' \
            .format(ground_truth['compiler_name'])

    _supported_compiler_types = ('clang', 'gcc', 'intel')
    base_compilers = {x: None for x in _supported_compiler_types}
    compiler_flags = {x: None for x in _supported_compiler_types}
    compiler_op_levels = {x: None for x in _supported_compiler_types}
    all_op_levels = {}
    all_switches = {}
    for compiler in projconf['compiler']:
        assert compiler['type'] in _supported_compiler_types, \
            'Unsupported compiler type: {}'.format(compiler['type'])
        assert base_compilers[compiler['type']] is None, \
            'You can only specify one of each type of compiler.'
        base_compilers[compiler['type']] = compiler['binary']
        
        switches = {flag_name(flag): flag for flag in compiler['switches_list']}
        compiler_flags[compiler['type']] = switches.keys()
        all_switches.update(switches)

        op_levels = {flag_name(flag): flag
                     for flag in compiler['optimization_levels']}
        compiler_op_levels[compiler['type']] = op_levels.keys()
        all_op_levels.update(op_levels)

    test_run_args = ''
    if not projconf['run']['timing']:
        test_run_args = '--no-timing'
    else:
        test_run_args = ' '.join([
            '--timing-loops', str(projconf['run']['timing_loops']),
            '--timing-repeats', str(projconf['run']['timing_repeats']),
            ])

    given_compilers = [key for key, val in base_compilers.items()
                       if val is not None]
    replacements = {
        'dev_compiler': matching_dev_compilers[0]['binary'],
        'dev_optl': dev_build['optimization_level'],
        'dev_switches': dev_build['switches'],
        'ground_truth_compiler': matching_gt_compilers[0]['binary'],
        'ground_truth_optl': ground_truth['optimization_level'],
        'ground_truth_switches': ground_truth['switches'],
        'flit_include_dir': conf.include_dir,
        'flit_lib_dir': conf.lib_dir,
        'flit_data_dir': conf.data_dir,
        'flit_script_dir': conf.script_dir,
        'flit_version': conf.version,
        'test_run_args': test_run_args,
        'enable_mpi': 'yes' if projconf['run']['enable_mpi'] else 'no',
        'mpirun_args': projconf['run']['mpirun_args'],
        'compilers': ' '.join([c.upper() for c in given_compilers]),
        'opcodes_definitions': generate_assignments(all_op_levels),
        'switches_definitions': generate_assignments(all_switches),
        }
    replacements.update({key + '_compiler': val
                         for key, val in base_compilers.items()})
    replacements.update({'opcodes_' + x: ' '.join(compiler_op_levels[x])
                         for x in given_compilers})
    replacements.update({'opcodes_' + x: None
                         for x in _supported_compiler_types
                         if x not in given_compilers})
    replacements.update({'switches_' + compiler: ' '.join(compiler_flags[compiler])
                         for compiler in given_compilers})
    replacements.update({'switches_' + x: None
                         for x in _supported_compiler_types
                         if x not in given_compilers})

    flitutil.process_in_file(os.path.join(conf.data_dir, 'Makefile.in'),
                             makefile, replacements, overwrite=True)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
