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

'Implements the import subcommand, importing results into a database'

import flitutil as util

import argparse
import csv
import datetime
import os
import sqlite3
import sys

brief_description = 'Import flit results into the configured database'

def _file_check(filename):
    'Check that a file exists or raise an exception'
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError('File does not exist: {0}'.format(filename))
    return filename

def get_dbfile_from_toml(tomlfile):
    'get and return the database.filepath field'
    import toml
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        return 1
    util.fill_defaults(projconf)

    assert projconf['database']['type'] == 'sqlite', \
            'Only sqlite database supported'
    return projconf['database']['filepath']

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Import flit results into the configured database.  The
                configured database is found from the settings in
                flit-config.toml.  You can import either exported results or
                results from manually running the tests.  Note that importing
                the same thing twice will result in having two copies of it
                in the database.
                ''',
            )
    parser.add_argument('importfile', nargs='+', type=_file_check,
                        help='''
                            File(s) to import into the database.  These files
                            may be csv files or sqlite3 databases.
                            ''')
    parser.add_argument('-a', '--append', type=int, default=None, metavar='RUN_ID',
                        help='''
                            Append the import to the specified run id.  The
                            default behavior is to add a new run to include the
                            results of the import.  You must specify a run id
                            that already exists in the database.
                            ''')
    parser.add_argument('-l', '--label', default='Imported using flit import',
                        help='''
                            The label to attach to the run.  Only applicable
                            when creating a new run.  This argument is ignored
                            if --append is specified.  The default label is
                            'Imported using flit import'.
                            ''')
    parser.add_argument('-r', '--run', type=int, default=None,
                        help='''
                            Only applicable to the importing of sqlite3
                            database files.  This will apply to all sqlite3
                            database files passed in.  Only this run id will be
                            imported from the provided database.  The default
                            behavior is to import the latest run.  You cannot
                            specify more than one run to be imported, you must
                            call this program multiple times, each one with
                            --run specified to the next run you want to import.
                            ''')
    parser.add_argument('-D', '--dbfile', default=None,
                        help='''
                            Use this database file rather than the one
                            specified in flit-config.toml.  This option is
                            especially useful when you want to import results
                            but do not have the flit-config.toml file
                            available, as that is currently the only reason for
                            flit-config.toml to be read by this command.  It
                            can also be used when you do not have the toml
                            python package installed (goodie!).
                            ''')
    args = parser.parse_args(arguments)

    if args.dbfile is None:
        args.dbfile = get_dbfile_from_toml('flit-config.toml')

    db = util.sqlite_open(args.dbfile)

    # create a new run and set the args.append run id
    if args.append is None:
        # Create a new run to use in import
        db.execute('insert into runs(rdate,label) values (?,?)',
                (datetime.datetime.now(), args.label))
        db.commit()
        args.append = db.execute('select id from runs order by id').fetchall()[-1]['id']

    # Make sure the run id exists.
    run_ids = [x['id'] for x in db.execute('select id from runs')]
    assert args.append in run_ids, \
            'Specified append run id {0} is not in the runs ' \
            'table'.format(args.append)

    for importee in args.importfile:
        print('Importing', importee)
        if util.is_sqlite(importee):
            import_db = util.sqlite_open(importee)
            cur = import_db.cursor()
            cur.execute('select id from runs')
            importee_run_ids = sorted([x['id'] for x in cur])
            if len(importee_run_ids) == 0:
                print('  no runs in database: nothing to import')
                continue
            latest_run = importee_run_ids[-1]
            import_run = args.run if args.run is not None else latest_run
            cur.execute('select name,host,compiler,optl,switches,precision,'
                        'comparison_hex,comparison,file,nanosec '
                        'from tests where run = ?', (import_run,))
            rows = [dict(x) for x in cur]
        else:
            with open(importee, 'r') as csvin:
                reader = csv.DictReader(csvin)
                rows = [row for row in reader]
        if len(rows) == 0:
            print('  zero rows: nothing to import')
            continue
        to_insert = []
        for row in rows:
            # Convert 'NULL' to None
            for key, val in row.items():
                row[key] = val if val != 'NULL' else None
            # Insert
            to_insert.append((
                args.append,
                row['name'],
                row['host'],
                row['compiler'],
                row['optl'],
                row['switches'],
                row['precision'],
                row['comparison_hex'],
                row['comparison'],
                row['file'],
                row['nanosec'],
                ))
        db.executemany('''
            insert into tests(
                run,
                name,
                host,
                compiler,
                optl,
                switches,
                precision,
                comparison_hex,
                comparison,
                file,
                nanosec)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', to_insert)
    db.commit()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
