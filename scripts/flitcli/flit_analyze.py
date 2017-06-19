'Implements the analyze subcommand'

import argparse
import sys

import flitconfig as conf

brief_description = 'Runs analysis on a previous flit run'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Runs analysis on a previous flit run.  The analysis will be of
                the current flit repository and will create a directory called
                analysis inside of the flit directory.
                ''',
            )
    parser.add_argument('-C', '--directory', default='.',
                        help='Directory containing flit.conf')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all runs that can be analyzed')
    parser.add_argument('run', nargs='*', default='latest',
                        help='''
                            Analyze the given run(s).  Defaults to the latest
                            run.
                            ''',)
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

    # TODO: read the configuration file to find the location of the database
    # TODO: allow the different types of analysis to be done


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
