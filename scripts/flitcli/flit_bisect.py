'''
Implements the bisect subcommand, identifying the problematic subset of source
files that cause the variability.
'''

import flitutil as util

import toml

import argparse
import csv
import datetime
import os
import sqlite3
import sys

brief_description = 'Bisect compilation to identify problematic source code'

def _file_check(filename):
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError('File does not exist: {0}'.format(filename))
    return filename

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Compiles the source code under both the ground-truth
                compilation and a given problematic compilation.  This tool
                then finds the minimal set of source files needed to be
                compiled under the problematic compilation flags so that the
                same answer is given.  This allows you to narrow down where the
                reproducibility problem lies.
                ''',
            )
    parser.add_argument('-c', '--compilation',
                        help='''
                            The problematic compilation to use.  This should
                            specify the compiler, optimization level, and
                            switches (which can be empty).  An example value
                            for this option would be "gcc -O2
                            -funsafe-math-optimizations" or
                            "/opt/intel/bin/icpc -O1".  The value will be split
                            into three groups using space separators, the first
                            is the compiler, the second is the optimization
                            level, and the third (if present) is the switches.
                            ''')
    args = parser.parse_args(arguments)

    tomlfile = 'flit-config.toml'
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        return 1

    # Split the compilation into the separate components
    compiler, optl, switches = args.compilation.strip().split(maxsplit=2)
    print('compiler: ', repr(compiler))
    print('optl:     ', repr(optl))
    print('switches: ', repr(switches))

    # TODO: see if the Makefile needs to be regenerated
    # TODO: get the list of source files from the Makefile
    # TODO: determine if the problem is on the linker's side
    # TODO:   I'm not yet sure the best way to do that
    # TODO: Perform in parallel binary search from ground-truth and from
    #       problematic
    # TODO: Use the custom comparison function in the test class when
    #       performing the binary search.
    # TODO: autogenerate Makefiles in the /tmp directly, preferrably with the
    #       tempfile module.

    # Below was copied from flit_import
    #assert projconf['database']['type'] == 'sqlite', \
    #        'Only sqlite database supported'
    #db = util.sqlite_open(projconf['database']['filepath'])

    ## create a new run and set the args.append run id
    #if args.append is None:
    #    # Create a new run to use in import
    #    db.execute('insert into runs(rdate,label) values (?,?)',
    #            (datetime.datetime.now(), args.label))
    #    db.commit()
    #    args.append = db.execute('select id from runs order by id').fetchall()[-1]['id']

    ## Make sure the run id exists.
    #run_ids = [x['id'] for x in db.execute('select id from runs')]
    #assert args.append in run_ids, \
    #        'Specified append run id {0} is not in the runs ' \
    #        'table'.format(args.append)

    #for importee in args.importfile:
    #    print('Importing', importee)
    #    if util.is_sqlite(importee):
    #        import_db = util.sqlite_open(importee)
    #        cur = import_db.cursor()
    #        cur.execute('select id from runs')
    #        importee_run_ids = sorted([x['id'] for x in cur])
    #        if len(importee_run_ids) == 0:
    #            print('  no runs in database: nothing to import')
    #            continue
    #        latest_run = importee_run_ids[-1]
    #        import_run = args.run if args.run is not None else latest_run
    #        cur.execute('select name,host,compiler,optl,switches,precision,'
    #                    'comparison,comparison_d,file,nanosec '
    #                    'from tests where run = ?', (import_run,))
    #        rows = [dict(x) for x in cur]
    #    else:
    #        with open(importee, 'r') as csvin:
    #            reader = csv.DictReader(csvin)
    #            rows = [row for row in reader]
    #    if len(rows) == 0:
    #        print('  zero rows: nothing to import')
    #        continue
    #    to_insert = []
    #    for row in rows:
    #        # Convert 'NULL' to None
    #        for key, val in row.items():
    #            row[key] = val if val != 'NULL' else None
    #        # Insert
    #        to_insert.append((
    #            args.append,
    #            row['name'],
    #            row['host'],
    #            row['compiler'],
    #            row['optl'],
    #            row['switches'],
    #            row['precision'],
    #            row['comparison'],
    #            row['comparison_d'],
    #            row['file'],
    #            row['nanosec'],
    #            ))
    #    db.executemany('''
    #        insert into tests(
    #            run,
    #            name,
    #            host,
    #            compiler,
    #            optl,
    #            switches,
    #            precision,
    #            comparison,
    #            comparison_d,
    #            file,
    #            nanosec)
    #        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #        ''', to_insert)
    #db.commit()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
