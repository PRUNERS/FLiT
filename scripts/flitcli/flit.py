#!/usr/bin/env python3
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

def main(arguments):
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

