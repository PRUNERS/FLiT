# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

'''
Utility functions shared between multiple flit subcommands.
'''

import flitconfig as conf

import os
import sqlite3
import sys

def process_in_file(infile, dest, vals, overwrite=False, remove_license=True):
    '''
    Process a file such as 'Makefile.in' where there are variables to
    replace.

    @param infile: input file.  Usually ends in ".in"
    @param dest: destination file.  If overwrite is False, then destination
        shouldn't exist, otherwise a warning is printed and nothing is
        done.
    @param vals: dictionary of key -> val where we search and replace {key}
        with val everywhere in the infile.
    @param overwrite: (default False) True means overwrite the destination.  If
        False, then a warning will be printed to the console and this function
        will return without doing anything.
    @param remove_license: (default True) True means remove the License
        declaration at the top of the file that has "-- LICENSE BEGIN --" at
        the beginning and "-- LICENSE END --" at the end.  All lines between
        including those lines will be removed.  False means ignore the license
        section.  If the license section is not there, then this will have no
        effect (except for a slight slowdown for searching)
    @return None
    '''
    if not overwrite and os.path.exists(dest):
        print('Warning: {0} already exists, not overwriting'.format(dest),
              file=sys.stderr)
        return
    with open(infile, 'r') as fin:
        if remove_license:
            fin_content = ''.join(remove_license_lines(fin))
        else:
            fin_content = fin.read()
        with open(dest, 'w') as fout:
            fout.write(fin_content.format(**vals))

def remove_license_lines(lines):
    '''
    Removes the license lines from the iterable of lines.  The license lines
    section will start with a line containing "-- LICENSE BEGIN --" and will
    end with a line containing "-- LICENSE END --".  All lines between the two
    including those two lines will be removed.

    @param lines: the iterable of lines to filter
    @return lines

    >>> remove_license_lines([
    ...     '-- something here --',
    ...     'bla bla bla -- LICENSE BEGIN --',
    ...     'my name is part of the license!',
    ...     '# -- LICENSE END -- **/',
    ...     'hello',
    ...     ])
    ['-- something here --', 'hello']

    >>> remove_license_lines([])
    []

    >>> remove_license_lines(['-- LICENSE BEGIN --', 'bla bla bla'])
    []

    >>> remove_license_lines(['bla bla bla', '-- LICENSE END --'])
    ['bla bla bla', '-- LICENSE END --']

    >>> lines = ['hello', '-- LICENSE BEGIN --', '-- LICENSE END --']
    >>> remove_license_lines(lines)
    ['hello']
    '''
    BEFORE = 0
    IN_LICENSE = 1
    AFTER = 2

    state = BEFORE
    filtered = []
    for line in lines:
        # state transitions
        if state == BEFORE and '-- LICENSE BEGIN --' in line:
            state = IN_LICENSE
            continue
        if state == IN_LICENSE and '-- LICENSE END --' in line:
            state = AFTER
            continue

        if state != IN_LICENSE:
            filtered.append(line)

    return filtered

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
