#!/usr/bin/env python3
'Extracts test run data into csv files'

import argparse
import csv
import pg
import sys

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

    pgdb = pg.DB(dbname='qfp', host='localhost', user='qfp', passwd='qfp123')
    runs = args.runs
    if len(runs) == 0:
        query_runs = pgdb.query('select index from runs;')
        idx = query_runs.dictresult()[-1]['index']
        runs.append(idx)

    for run in runs:
        filename = 'run-{0:02}.csv'.format(run)
        sys.stdout.write('writing ' + filename + ' ...')
        sys.stdout.flush()
        query_tests = pgdb.query('select * from tests where run = $1;', run)
        query_results_to_file(filename, query_tests)
        sys.stdout.write(' done\n')
        sys.stdout.flush()
        #print(query_tests)

if __name__ == '__main__':
    main(sys.argv[1:])

