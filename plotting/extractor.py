#!/usr/bin/env python3
'Extracts test run data into csv files'

import argparse
import csv
import sys

try:
    import pg
    using_pg = True
except ImportError:
    import psycopg2
    using_pg = False

def connect_to_database():
    'Returns a database handle for the framework being used'
    dbname = 'qfp'
    host = 'localhost'
    user = 'qfp'
    passwd = 'qfp123'
    if using_pg:
        return pg.DB(dbname=dbname, host=host, user=user, passwd=passwd)
    else:
        conn_string = ''
        conn = psycopg2.connect(conn_string)
        return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

def run_query(handle, query, *args, **kwargs):
    'Runs the query and returns the result as a list of dictionaries'
    if using_pg:
        return handle.query(query, *args, **kwargs).dictresults()
    else:
        handle.execute(query, *args, **kwargs)
        return handle.fetchall()

def query_results_to_file(filename, query):
    'Writes results from a PyGresQL query object to a csv file.'
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, query.listfields())
        writer.writeheader()
        writer.writerows(query.dictresult())

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        Extracts test run data into csv files. Saves runs to run-N.csv.
        The default behavior is to only extract the most recent run.
        ''')
    parser.add_argument('runs', metavar='N', type=int, nargs='*',
                        help='A run to extract to csv.  Saved as run-N.csv')
    args = parser.parse_args(args=arguments)

    db = connect_to_database()
    runs = args.runs
    if len(runs) == 0:
        query_runs = run_query(db, 'select index from runs;')
        idx = query_runs.dictresult()[-1]['index']
        runs.append(idx)

    for run in runs:
        filename = 'run-{0:02}.csv'.format(run)
        sys.stdout.write('writing ' + filename + ' ...')
        sys.stdout.flush()
        query_tests = run_query(db, 'select * from tests where run = $1;', run)
        query_results_to_file(filename, query_tests)
        sys.stdout.write(' done\n')
        sys.stdout.flush()
        #print(query_tests)

if __name__ == '__main__':
    main(sys.argv[1:])

