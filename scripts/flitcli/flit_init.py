'Implements the init subcommand'

import argparse
import os
import shutil
import socket
import sys
import toml

import flitconfig as conf
import flitutil
import flit_update

brief_description = 'Initializes a flit test directory for use'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
        prog=prog,
        description='''
            Initializes a flit test directory for use.  It will initialize
            the directory by copying the default configuration file into
            the given directory.  If a configuration file already exists,
            this command does nothing.  The config file is called
            flit-config.toml.
            ''',
        )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite init files if they are already there')
    parser.add_argument('-L', '--litmus-tests', action='store_true',
                        help='Copy over litmus tests too')
    args = parser.parse_args(arguments)

    os.makedirs(args.directory, exist_ok=True)

    # write flit-config.toml
    flit_config_dest = os.path.join(args.directory, 'flit-config.toml')
    print('Creating {0}'.format(flit_config_dest))
    flitutil.process_in_file(
        os.path.join(conf.config_dir, 'flit-default.toml.in'),
        flit_config_dest,
        {
            'flit_path': os.path.join(conf.script_dir, 'flit.py'),
            'config_dir': conf.config_dir,
            'hostname': socket.gethostname(),
        },
        overwrite=args.overwrite)

    # Copy the remaining files over
    to_copy = {
        'custom.mk': os.path.join(conf.data_dir, 'custom.mk'),
        'main.cpp': os.path.join(conf.data_dir, 'main.cpp'),
        'tests/Empty.cpp': os.path.join(conf.data_dir, 'tests/Empty.cpp'),
        }

    # Add litmus tests too
    if args.litmus_tests:
        for srcfile in os.listdir(conf.litmus_test_dir):
            if os.path.splitext(srcfile)[1] in ('.cpp', '.h'):
                srcpath = os.path.join(conf.litmus_test_dir, srcfile)
                to_copy[os.path.join('tests', srcfile)] = srcpath

    for dest, src in to_copy.items():
        realdest = os.path.join(args.directory, dest)
        print('Creating {0}'.format(realdest))
        if not args.overwrite and os.path.exists(realdest):
            print('Warning: {0} already exists, not overwriting'.format(realdest),
                  file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(os.path.realpath(realdest)), exist_ok=True)
        shutil.copy(src, realdest)

    flit_update.main(['--directory', args.directory])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
