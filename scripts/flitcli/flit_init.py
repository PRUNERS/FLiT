'Implements the init subcommand'

import argparse
import os
import shutil
import sys
import toml

import flitconfig as conf

brief_description = 'Initializes a flit test directory for use'

def process_in_file(infile, dest, vals, overwrite=False):
    '''
    Process a file such as 'Makefile.in' where there are variables to
    replace.

    @param infile: input file.  Usually ends in ".in"
    @param dest: destination file.  If overwrite is False, then destination
        shouldn't exist, otherwise a warning is printed and nothing is
        done.
    @param vals: dictionary of key -> val where we search and replace {key}
        with val everywhere in the infile.
    '''
    if not overwrite and os.path.exists(dest):
        print('Warning: {0} already exists, not overwriting'.format(dest))
    else:
        with open(infile, 'r') as fin:
            with open(dest, 'w') as fout:
                fout.write(fin.read().format(**vals))

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
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite init files if they are already there')
    args = parser.parse_args(arguments)

    os.makedirs(args.directory, exist_ok=True)

    # write flit-config.toml
    flit_config_dest = os.path.join(args.directory, 'flit-config.toml')
    print('Creating {0}'.format(flit_config_dest))
    process_in_file(
        os.path.join(conf.config_dir, 'flit-default.toml.in'),
        flit_config_dest,
        {
            'flit_path': os.path.join(conf.script_dir, 'flit.py').__repr__(),
            'config_dir': conf.config_dir.__repr__(),
        },
        overwrite=args.overwrite)

    projconf = toml.load(flit_config_dest)
    print('Creating {0}'.format(os.path.join(args.directory, 'Makefile')))
    process_in_file(
        os.path.join(conf.data_dir, 'Makefile.in'),
        os.path.join(args.directory, 'Makefile'),
        {
            'compiler': os.path.realpath(projconf['hosts'][0]['compilers'][0]['binary']),
            'flit_include_dir': conf.include_dir,
            'flit_lib_dir': conf.lib_dir,
            'flit_script': os.path.join(conf.script_dir, 'flit.py'),
        },
        overwrite=args.overwrite)

    # Copy the remaining files over
    to_copy = {
        'custom.mk': os.path.join(conf.data_dir, 'custom.mk'),
        'main.cpp': os.path.join(conf.data_dir, 'main.cpp'),
        'tests/Empty.cpp': os.path.join(conf.data_dir, 'tests/Empty.cpp'),
        }

    for dest, src in to_copy.items():
        realdest = os.path.join(args.directory, dest)
        print('Creating {0}'.format(realdest))
        if not args.overwrite and os.path.exists(realdest):
            print('Warning: {0} already exists, not overwriting'.format(realdest))
            continue
        os.makedirs(os.path.dirname(os.path.realpath(realdest)), exist_ok=True)
        shutil.copy(src, realdest)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
