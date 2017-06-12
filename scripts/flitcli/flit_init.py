'Implements the init subcommand'

import argparse
import os
import shutil
import sys

import flitconfig as conf

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
    # TODO: add argument to overwrite files
    #parser.add_argument('--overwrite', action='store_true',
    #                    help='Overwrite init files if they are already there')
    args = parser.parse_args(arguments)

    os.makedirs(args.directory, mode=0o755, exist_ok=True)
    os.chdir(args.directory)

    # Copy the data directory contents and the default configuration
    # TODO: these Makefiles should actually be made with "flit update"
    to_copy = {x: os.path.join(conf.data_dir, x) for x in os.listdir(conf.data_dir)}
    # TODO: uncomment when flit-config.toml is actually used
    #to_copy['flit-config.toml'] = os.path.join(conf.config_dir, 'flit-default.toml')

    for dest, src in to_copy.items():
        if os.path.exists(dest):
            print('Warning: {0} already exists, not overwriting'.format(dest))
            continue
        print('Creating {0}'.format(dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
