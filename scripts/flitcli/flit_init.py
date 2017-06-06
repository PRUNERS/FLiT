'Implements the init subcommand'

import argparse
import sys

brief_description = 'Initializes a flit test directory for use'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Initializes a flit test directory for use.  It will initialize
                the directory by copying the default configuration file into
                the given directory.  If a configuration file already exists,
                this command does nothing.  The config file is called
                flit.conf.
                ''',
            )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

    # TODO: find the default configuration file.  Its path should be in
    #       flitconfig.py
    # TODO: create the directory if it doesn't exist
    # TODO: copy the default configuration file if it doesn't exist

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
