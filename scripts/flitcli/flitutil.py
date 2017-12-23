'''
Utility functions shared between multiple flit subcommands.
'''

import flitconfig as conf

import os
import sqlite3
import subprocess as subp
import sys
import tempfile

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
        print('Warning: {0} already exists, not overwriting'.format(dest),
              file=sys.stderr)
        return
    with open(infile, 'r') as fin:
        with open(dest, 'w') as fout:
            fout.write(fin.read().format(**vals))

def sqlite_open(filepath):
    '''
    Opens and returns an sqlite database cursor object.  If the database does
    not exist, it will be created.
    '''
    # Using detect_types allows us to insert datetime objects
    connection = sqlite3.connect(filepath,
                                 detect_types=sqlite3.PARSE_DECLTYPES)

    # Use the dict factory so that queries return dictionary-like objects
    connection.row_factory = sqlite3.Row

    # Create the tables if they do not exist.  Also has other setup.
    table_file = os.path.join(conf.data_dir, 'db', 'tables-sqlite.sql')
    with open(table_file, 'r') as table_sql:
        connection.executescript(table_sql.read())
    connection.commit()

    return connection

def is_sqlite(filename):
    'Returns true if the file is likely an sqlite file.'
    from os.path import isfile, getsize

    if not os.path.isfile(filename):
        return False

    # SQLite database file header is 100 bytes
    if os.path.getsize(filename) < 100:
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[:16] == b'SQLite format 3\000'

def extract_make_var(var, makefile='Makefile'):
    '''
    Extracts the value of a particular variable within a particular Makefile.

    How it works with a valid file:

    >>> from tempfile import NamedTemporaryFile as NTF
    >>> with NTF(mode='w+') as fout:
    ...     print('A    := hello there sweetheart\\n', file=fout, flush=True)
    ...     A = extract_make_var('A', fout.name)
    >>> A
    ['hello', 'there', 'sweetheart']

    If the variable is undefined, then simply an empty list is returned.

    >>> with NTF() as fout: extract_make_var('A', fout.name)
    []

    What if the file does not exist?  It throws an exception:

    >>> extract_make_var('A', 'file-should-not-exist.mk') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    subprocess.CalledProcessError: Command ... returned non-zero exit status 2.
    '''
    with tempfile.NamedTemporaryFile(mode='w+') as fout:
        print('print-%:\n'
              "\t@echo '$*=$($*)'\n", file=fout, flush=True)
        output = subp.check_output(['make', '-f', makefile, '-f', fout.name,
                                    'print-' + var], stderr=subp.STDOUT)
    output = output.strip().decode('utf-8')
    var_values = output.split('=', maxsplit=1)[1].split()
    return var_values

