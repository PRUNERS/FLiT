'Implements the squelch subcommand'

import argparse
import sys

brief_description = 'Run flit on the configured remote machine(s)'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Run flit on the configured remote machine(s).  Note that you
                may need to provide a password for SSH, but that should be
                taken care of pretty early on in the process.  The results
                should be sent to the database computer for later analysis.
                ''',
            )
    parser.add_argument('directory', default='.',
                        help='The directory to initialize')
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

    # TODO: execute flit make on the remote machines
    # TODO: push the results to the database machine

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
