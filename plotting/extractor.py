#!/usr/bin/env python
'Extracts test run data into csv files'

from __future__ import print_function

import argparse
import csv
import sys

try:
    import pg
    using_pg = True
except ImportError:
    import psycopg2
    import psycopg2.extras
    using_pg = False

def connect_to_database():
    'Returns a database handle for the framework being used'
    dbname = 'flit'
    host = 'localhost'
    user = 'flit'
    passwd = 'flit123'
    if using_pg:
        return pg.DB(dbname=dbname, host=host, user=user, passwd=passwd)
    else:
        conn_string = "dbname='{dbname}' user='{user}' host='{host}' password='{passwd}'" \
                .format(dbname=dbname, host=host, user=user, passwd=passwd)
        conn = psycopg2.connect(conn_string)
        return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

def run_query(handle, query, *args, **kwargs):
    'Runs the query and returns the result as a list of dictionaries'
    if using_pg:
        return handle.query(query, *args, **kwargs).dictresults()
    else:
        handle.execute(query, *args, **kwargs)
        return [dict(x) for x in handle.fetchall()]

def query_results_to_file(filename, query):
    'Writes results from a PyGresQL query object to a csv file.'
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, sorted(query[0].keys()))
        writer.writeheader()
        writer.writerows(query)

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        Extracts test run data into csv files. Saves runs to run-N.csv.
        The default behavior is to only extract the most recent run.
        ''')
    parser.add_argument('runs', metavar='N', type=int, nargs='*',
                        help='A run to extract to csv.  Saved as run-N.csv')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List the avilable runs for download')
    args = parser.parse_args(args=arguments)

    db = connect_to_database()
    runs = args.runs
    if len(runs) == 0:
        query_runs = run_query(db, 'select index from runs;')
        idx = query_runs[-1]['index']
        runs.append(idx)
        if args.list:
            print('\n'.join([str(x['index']) for x in query_runs]))
            return 0

    for run in runs:
        filename = 'run-{0:02}.csv'.format(run)
        sys.stdout.write('writing ' + filename + ' ...')
        sys.stdout.flush()
        query_tests = run_query(db, 'select * from tests where run = \'{0}\';'.format(run))
        query_results_to_file(filename, query_tests)
        sys.stdout.write(' done\n')
        sys.stdout.flush()
        #print(query_tests)
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])

