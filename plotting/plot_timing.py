#!/usr/bin/env python3
'''
Plots the speedup vs compilation.  The x-axis is organized to make the speedup
curve be monotonically non-decreasing.  Along the speedup curve, there are
marks at each point.  A blue dot represents where the answer was the same as
the ground truth.  A red X represents where the answer differed from the ground
truth answer.
'''

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
import sys

def read_csv(csvfile):
    '''
    Reads and returns the csvfile as a list of dictionaries
    '''
    rows = []
    with open(csvfile, 'r') as fin:
        reader = csv.DictReader(fin)
        rows = [row for row in reader]
    return rows

def read_sqlite(sqlitefile):
    connection = sqlite3.connect(sqlitefile,
            detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    cur = connection.cursor()
    cur.execute('select id from runs order by id')
    run_ids = sorted([x['id'] for x in cur])
    if len(run_ids) == 0:
        raise RuntimeError('No runs in the database: ' + sqlitefile)
    latest_run = run_ids[-1]
    read_run = latest_run
    cur.execute('select * from tests where run = ?', (read_run,))
    return [dict(x) for x in cur]

def plot_timing(csvfile, test_names):
    '''
    Plots the timing metrics from the csv file and for the given test_names
    '''
    rows = read_csv(csvfile)
    to_x_lab = lambda row: '{0} {1} {2}'.format(row['compiler'], row['optl'], row['switches'])
    test_data = {}
    for name in test_names:
        data = {}
        data['name'] = name
        data['rows'] = [row for row in rows
                        if row['name'] == name
                        and row['precision'] == 'f'
                        and (row['optl'] != '-O0' or row['switches'] == '')
                        ]
        data['times'] = np.asarray([int(row['nanosec']) for row in data['rows']])
        data['fastest'] = min(data['times'])
        data['slowest'] = max(data['times'])
        data['speedup'] = data['slowest'] / data['times']
        data['xlab'] = [to_x_lab(row) for row in data['rows']]
        data['unopt'] = [row['score0'] for row in data['rows']
                if row['optl'] == '-O0' and row['switches'] == '']
        assert all([x == data['unopt'][0] for x in data['unopt']]), str(data['unopt'])
        data['unopt'] = data['unopt'][0]
        data['iseql'] = [row['score0'] == data['unopt'] for row in data['rows']]
        test_data[name] = data
    for name in test_data:
        print(name, max(test_data[name]['speedup']))
        print('  slowest', test_data[name]['slowest'])
        print('  fastest', test_data[name]['fastest'])
        speedup = test_data[name]['speedup']
        xlab = test_data[name]['xlab']
        iseql = test_data[name]['iseql']
        joint_sort = sorted(zip(xlab, speedup, iseql), key=lambda x: x[1])
        xlab = [x[0] for x in joint_sort]
        speedup = [x[1] for x in joint_sort]
        iseql = [x[2] for x in joint_sort]
        eql_idxs = [i for i in range(len(iseql)) if iseql[i] is True]
        not_eql_idxs = [i for i in range(len(iseql)) if iseql[i] is False]
        eql_speeds = [speedup[i] for i in eql_idxs]
        not_eql_speeds = [speedup[i] for i in not_eql_idxs]

        plt.figure(num=1, figsize=(12.5,5), dpi=80)
        plt.plot(speedup)#, label=name)
        plt.plot(eql_idxs, eql_speeds, 'b.', label='same answer as unoptimized')
        plt.plot(not_eql_idxs, not_eql_speeds, 'rx', label='different answer than unoptimized')
        plt.xticks(np.arange(len(xlab)), xlab, rotation='vertical')
        plt.legend()
        plt.ylim(ymin=0)
        plt.ylabel('Speedup from unoptimized')
        plt.tight_layout()
        #plt.show()
        newname = name + '.svg'
        plt.savefig(newname, format='svg')
        plt.cla()
    #fig, ax = plt.subplots()
    #plt.xticks(np.arange(len(flags)), flags, rotation='vertical')
    #plt.show()
    #base = os.path.basename(csvfile)
    #newname = os.path.splitext(base)[0] + '.svg'
    #plt.savefig(newname, format='svg', dpi=300)


def main(arguments):
    'Main entry point, calls plot_timing()'
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('test', nargs='+')
    args = parser.parse_args(arguments)
    plot_timing(args.csv, args.test)

if __name__ == '__main__':
    main(sys.argv[1:])
