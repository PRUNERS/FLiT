#!/usr/bin/env python3

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
from collections import namedtuple

import flitargformatter
import flitconfig as conf

Subcommand = namedtuple('Subcommand',
                        'name, brief_description, main, populate_parser')

def load_subcommands(directory):
    '''
    Creates subcommands from modules found in the given directory.

    From that directory, all subcommands will be loaded with the pattern
    flit_*.py.

    Returns a list of Subcommand tuples.
    '''
    if directory not in sys.path:
        sys.path.insert(0, directory)
    subcommand_files = glob.glob(os.path.join(directory, 'flit_*.py'))
    subcommand_names = [os.path.basename(x)[5:-3] for x in subcommand_files]
    subcom_modules = [importlib.import_module(os.path.basename(x)[:-3])
                      for x in subcommand_files]
    module_map = dict(zip(subcommand_names, subcom_modules))

    # Make sure each module has the expected interface
    for name, module in module_map.items():
        expected_attributes = [
            'brief_description',
            'populate_parser',
            'main',
            ]
        for attr in expected_attributes:
            assert hasattr(module, attr), \
                'Module flit_{0} is missing attribute {1}'.format(name, attr)

    subcommands = [
        Subcommand(name, m.brief_description, m.main, m.populate_parser)
        for name, m in module_map.items()]

    return subcommands

def populate_parser(parser=None, subcommands=None, recursive=False):
    '''
    Populate and return the ArgumentParser.  If not given, a new one is made.
    All arguments are optional.

    @param parser (argparse.ArgumentParser): parser to populate or None to
        generate a new one. (default=None)
    @param subcommands (list(Subcommand)): list of subcommands (default=None)
    @param recursive (bool): True means to add subcommand parsing beyond
        the top-level
    '''
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.formatter_class = flitargformatter.DefaultsParaHelpFormatter
    parser.description = '''
            The flit command-line tool allows for users to write portability
            test cases.  One can test primarily for compiler effects on
            reproducibility of floating-point algorithms.  That at least is the
            main use case for this tool, although you may come up with some
            other uses.
            '''
    parser.add_argument('-v', '--version', action='version',
                        version='flit version ' + conf.version,
                        help='Print version and exit')
    if subcommands:
        subparsers = parser.add_subparsers(
            title='Subcommands',
            dest='subcommand',
            metavar='subcommand')
        for subcommand in subcommands:
            subparser = subparsers.add_parser(
                subcommand.name, help=subcommand.brief_description,
                add_help=recursive)
            if recursive:
                subcommand.populate_parser(subparser)
    return parser

def main(arguments, module_dir=conf.script_dir, outstream=None,
         errstream=None, prog=None):
    '''
    Main logic here.

    For ease of use when within python, the stdout can be captured using the
    optional outstream parameter.  You can use this to capture the stdout that
    would go to the console and put it into a StringStream or maybe a file.
    '''
    if outstream is None and errstream is None:
        return _main_impl(arguments, module_dir, prog=prog)
    oldout = sys.stdout
    olderr = sys.stderr
    try:
        if outstream is not None:
            sys.stdout = outstream
        if errstream is not None:
            sys.stderr = errstream
        return _main_impl(arguments, module_dir, prog=prog)
    finally:
        sys.stdout = oldout
        sys.stderr = olderr

def create_help_subcommand(subcommands):
    'Create and return the help subcommand'
    subcommand_map = {sub.name: sub for sub in subcommands}

    help_description = 'display help for a specific subcommand'

    def help_populate_parser(parser=None):
        'populate_parser() for the help subcommand'
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.description = help_description
        subparsers = parser.add_subparsers(
            title='Subcommands',
            dest='help_subcommand',
            metavar='subcommand')
        for subcommand in subcommand_map.values():
            subparsers.add_parser(subcommand.name,
                                  help=subcommand.brief_description)
        return parser

    help_parser = help_populate_parser()

    def help_main(arguments, prog=sys.argv[0]):
        'main() for the help subcommand'
        help_parser.prog = prog
        args = help_parser.parse_args(arguments)
        if args.help_subcommand:
            sub = subcommand_map[args.help_subcommand].populate_parser()
            sub.print_help()
        else:
            help_parser.print_help()
        return 0

    return Subcommand('help', help_description, help_main,
                      help_populate_parser)

def _main_impl(arguments, module_dir, prog=None):
    'Implementation of main'
    subcommands = load_subcommands(module_dir)
    subcommands.append(create_help_subcommand(subcommands))

    parser = populate_parser(subcommands=subcommands)
    if prog: parser.prog = prog
    args, remaining = parser.parse_known_args(arguments)

    subcommand_map = {sub.name: sub for sub in subcommands}
    subcommand = subcommand_map[args.subcommand]
    return subcommand.main(remaining, prog=parser.prog + ' ' + args.subcommand)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
