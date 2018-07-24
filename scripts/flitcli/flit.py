#!/usr/bin/env python3

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
This script simply forwards commands on to the runner scripts.  For example,
for the subcommand "init", this script will import the module flit_init.py from
the same directory as this script and will forward all arguments to that
script.  Think of this script as a proxy in that way.  So instead of calling
"flit_init ...", you would call "flit init ...".  That hopefully seems more
natural.
'''


import argparse
import glob
import importlib
import os
import sys

def import_helper_modules(directory):
    'Imports the modules found in the given directory.'
    if directory not in sys.path:
        sys.path.insert(0, directory)
    subcommand_files = glob.glob(os.path.join(directory, 'flit_*.py'))
    subcommands = [os.path.basename(x)[5:-3] for x in subcommand_files]
    subcom_modules = [importlib.import_module(os.path.basename(x)[:-3])
                      for x in subcommand_files]
    subcom_map = dict(zip(subcommands, subcom_modules))

    # Make sure each module has the expected interface
    for name, module in subcom_map.items():
        expected_attributes = [
            'brief_description',
            'main',
            ]
        for attr in expected_attributes:
            assert hasattr(module, attr), \
                'Module {0} is missing attribute {1}'.format(name, attr)

    return subcom_map

def generate_help_documentation(subcom_map):
    '''
    Generates and returns both the formatted help for the general flit
    executable, but also for the help subcommand.  They are returned as a
    tuple.

    >>> help_str, help_subcom_str = generate_help_documentation(dict())
    '''
    parser = argparse.ArgumentParser(
            description='''
                The flit command-line tool allows for users to write
                portability test cases.  One can test primarily for
                compiler effects on reproducibility of floating-point
                algorithms.  That at least is the main use case for this
                tool, although you may come up with some other uses.
                ''',
            )
    parser.add_argument('-v', '--version', action='store_true',
                        help='Print version and exit')
    subparsers = parser.add_subparsers(metavar='subcommand', dest='subcommand')
    help_subparser = subparsers.add_parser(
            'help', help='display help for a specific subcommand')
    help_subparser.add_argument(
            metavar='subcommand',
            dest='help_subcommand',
            choices=subcom_map.keys(),
            help='''
                display the help documentation for a specific subcommand.
                choices are {0}.
                '''.format(', '.join(sorted(subcom_map.keys()))),
            )
    for name, module in sorted(subcom_map.items()):
        subparsers.add_parser(name, help=module.brief_description)

    # Note: we do not use parser for the actual parsing, because we want the
    # arguments for each subcommand to be handled by the associated module.
    # That does not seem to be well supported by the argparse module.

    return (parser.format_help(), help_subparser.format_help())

def main(arguments, outstream=None):
    '''
    Main logic here.

    For ease of use when within python, the stdout can be captured using the
    optional outstream parameter.  You can use this to capture the stdout that
    would go to the console and put it into a StringStream or maybe a file.
    '''
    if outstream is None:
        return _main_impl(arguments)
    else:
        try:
            oldout = sys.stdout
            sys.stdout = outstream
            _main_impl(arguments)
        finally:
            sys.stdout = oldout

def _main_impl(arguments):
    'Implementation of main'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, script_dir)
    import flitconfig as conf

    subcom_map = import_helper_modules(script_dir)

    help_str, help_subcommand_str = generate_help_documentation(subcom_map)
    if len(arguments) == 0 or arguments[0] in ('-h', '--help'):
        print(help_str)
        return 0

    if arguments[0] in ('-v', '--version'):
        print('flit version', conf.version)
        return 0

    all_subcommands = ['help'] + list(subcom_map.keys())
    subcommand = arguments.pop(0)

    if subcommand not in all_subcommands:
        sys.stderr.write('Error: invalid subcommand: {0}.\n' \
                         .format(subcommand))
        sys.stderr.write('Call with --help for more information\n')
        return 1

    if subcommand == 'help':
        if len(arguments) == 0:
            help_subcommand = 'help'
        else:
            help_subcommand = arguments.pop(0)

        if help_subcommand in ('-h', '--help', 'help'):
            print(help_subcommand_str)
            return 0

        elif help_subcommand not in all_subcommands:
            sys.stderr.write('Error: invalid subcommand: {0}.\n' \
                             .format(subcommand))
            sys.stderr.write('Call with --help for more information\n')
            return 1

        else:
            # just forward to the documentation from the submodule
            return subcom_map[help_subcommand].main(
                    ['--help'], prog='{0} {1}'.format(sys.argv[0], help_subcommand))
    else:
        # it is one of the other subcommands.  Just forward the request on
        return subcom_map[subcommand].main(
                arguments, prog='{0} {1}'.format(sys.argv[0], subcommand))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

