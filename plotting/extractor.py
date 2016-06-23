#!/usr/bin/env python3
'Extracts test run data into csv files'

import csv
import pg

def query_results_to_file(filename, query):
    'Writes results from a PyGresQL query object to a csv file.'
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, query.listfields())
        writer.writeheader()
        writer.writerows(query.dictresult())

def main():
    'Main entry point'
    db = pg.DB(dbname='qfp', host='localhost', user='qfp', passwd='qfp123')
    query_runs = db.query('select index from runs;')
    idx = query_runs.dictresult()[-1]['index']
    query_tests = db.query('select * from tests where run = $1;', idx)
    query_results_to_file('latest-run.csv', query_tests)
    #print(query_tests)

if __name__ == '__main__':
    main()

