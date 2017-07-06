'Implements the import subcommand, importing results into a database'

import flitutil as util

import toml

import argparse
import csv
import datetime
import os
import sqlite3
import sys

brief_description = 'Import flit results into the configured database'

def _file_check(filename):
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError('File does not exist: {0}'.format(filename))
    return filename

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
    args = parser.parse_args(arguments)

    try:
        projconf = toml.load('flit-config.toml')
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        return 1

    assert projconf['database']['type'] == 'sqlite', \
            'Only sqlite database supported'
    db = util.sqlite_open(projconf['database']['filepath'])

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
                        'comparison,comparison_d,file,nanosec '
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
                row['comparison'],
                row['comparison_d'],
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
                comparison,
                comparison_d,
                file,
                nanosec)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', to_insert)
    db.commit()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
