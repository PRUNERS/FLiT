'Implements the check subcommand'

import argparse
import sys

brief_description = 'Verifies the correctness of a config file'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                This command only verifies the correctness of the
                configurations you have for your flit tests.  As part of this
                verification, this command checks to see if the remote
                connections are capable of being done, such as the connection
                to the machines to run the software, the connection to the
                database machine, and the connection to the database machine
                from the run machine.  You may need to provide a few SSH
                passwords to do this check.
                ''',
            )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory containing flit.conf')
    parser.add_argument('-c', '--config-only', action='store_true',
                        help='''
                            Only check the config file instead of also the SSH
                            connections and maybe other things.''')
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

    # TODO: Check the configuration file
    # TODO: Check SSH connections

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
