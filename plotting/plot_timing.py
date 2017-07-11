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
import numpy as np
import os
import sqlite3
import sys

# This matplot command makes the use of pyplot without X11 possible
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_csv(csvfile):
    '''
    Reads and returns the csvfile as a list of dictionaries
    '''
    rows = []
    with open(csvfile, 'r') as fin:
        reader = csv.DictReader(fin)
        rows = [row for row in reader]
    return rows

def read_sqlite(sqlitefile, run=None):
    connection = sqlite3.connect(sqlitefile,
            detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    cur = connection.cursor()
    cur.execute('select id from runs order by id')
    run_ids = sorted([x['id'] for x in cur])
    if len(run_ids) == 0:
        raise RuntimeError('No runs in the database: ' + sqlitefile)
    if run is None:
        run = run_ids[-1]
    else:
        assert run in run_ids
    cur.execute('select * from tests where run = ?', (run,))
    return [dict(x) for x in cur]

def plot_timing(rows, test_names=[], outdir='.'):
    '''
    Plots the timing metrics from the rows and for the given test_names.  The
    resultant plots will be placed in outdir.

    If test_names is empty, then all tests in the rows will be plotted.
    '''
    # Make sure test_names are found in the rows.
    # Also, if test_names is empty, plot all tests
    all_test_names = set(x['name'] for x in rows)
    if len(test_names) == 0:
        test_names = sorted(all_test_names)
    assert all(x in all_test_names for x in test_names), \
            'unfound test names detected'
    
    # Make sure outdir exists
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    # generates labels for the x-axis
    to_x_label = lambda row: \
            '{0} {1} {2}' \
            .format(row['compiler'], row['optl'], row['switches']) \
            .strip()

    test_data = {}
    for name in test_names:
        test_rows = [row for row in rows if row['name'] == name]
        hosts = set(row['host'] for row in test_rows)
        for host in hosts:
            host_rows = [row for row in test_rows if row['host'] == host]
            precisions = set(row['precision'] for row in host_rows)
            for p in precisions:
                p_rows = [row for row in host_rows if row['precision'] == p]
                data = {}
                data['name'] = name
                data['rows'] = p_rows
                data['times'] = np.asarray([int(row['nanosec'])
                                            for row in data['rows']])
                data['fastest'] = min(data['times'])
                data['slowest'] = max(data['times'])
                # TODO: instead of calculating the speedup using the slowest
                # TODO: time, use the ground-truth time.
                data['speedup'] = data['slowest'] / data['times']
                data['xlab'] = [to_x_label(row) for row in data['rows']]
                data['iseql'] = [float(row['comparison_d']) == 0.0
                                 for row in data['rows']]
                key = (name, host, p)
                test_data[key] = data

    for key, data in test_data.items():
        name, host, p = key
        print(name, max(data['speedup']))
        print('  slowest', data['slowest'])
        print('  fastest', data['fastest'])
        speedup = data['speedup']
        xlab = data['xlab']
        iseql = data['iseql']
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
        plt.plot(eql_idxs, eql_speeds, 'b.',
                 label='same answer as ground truth')
        plt.plot(not_eql_idxs, not_eql_speeds, 'rx',
                 label='different answer than ground truth')
        plt.xticks(np.arange(len(xlab)), xlab, rotation='vertical')
        plt.legend()
        plt.ylim(ymin=0)
        plt.ylabel('Speedup from slowest')
        plt.tight_layout()
        #plt.show()
        newname = '{0}-{1}-{2}.svg'.format(name, host, p)
        plt.savefig(os.path.join(outdir, newname), format='svg')
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
    parser.add_argument('-o', '--outdir', default='.',
            help='Specify output directory for generated plots')
    parser.add_argument('-r', '--run', default=None, type=int,
            help='Which run to use from the sqlite database')
    parser.add_argument('-p', '--precision', default='all',
            choices=['all', 'f', 'd', 'e'],
            help='Which precision to draw.  By default does all precisions')
    #parser.add_argument('csv')
    parser.add_argument('sqlite')
    parser.add_argument('test', nargs='*')
    args = parser.parse_args(arguments)
    #rows = read_csv(args.csv)
    rows = read_sqlite(args.sqlite, args.run)
    if args.precision != 'all':
        rows = [x for x in rows if x['precision'] == args.precision]
    plot_timing(rows, args.test, args.outdir)

if __name__ == '__main__':
    main(sys.argv[1:])
