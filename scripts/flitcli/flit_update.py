'Implements the make subcommand'

import argparse
import sys

brief_description = 'Compiles and runs the flit tests locally'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                ''',
            )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    parser.add_argument('-s', '--simple', action='store_true',
                        help='''
                            Perform the simple run of a single compilation
                            rather than the full run
                            ''')
    parser.add_argument('-n', '--no-push', action='store_true',
                        help='Do not push to the database, just run locally')
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

    # TODO: Run the makefile
    # TODO: Push the results to the database

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
