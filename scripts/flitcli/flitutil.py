'''
Utility functions shared between multiple flit subcommands.
'''

import os

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


