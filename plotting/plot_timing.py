#!/usr/bin/env python3

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
        os.makedirs(outdir)
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
                # TODO- time, use the ground-truth time.
                data['speedup'] = data['slowest'] / data['times']
                data['xlab'] = [to_x_label(row) for row in data['rows']]
                data['iseql'] = [float(row['comparison']) == 0.0
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

        plt.figure(num=1, figsize=(3 + 0.13*len(speedup), 5), dpi=80)
        plt.plot(speedup)
        plt.plot(eql_idxs, eql_speeds, 'b.',
                 label='same answer as ground truth')
        plt.plot(not_eql_idxs, not_eql_speeds, 'rx',
                 label='different answer than ground truth')
        plt.xticks(np.arange(len(xlab)), xlab, rotation='vertical')
        plt.legend()
        plt.ylim(ymin=0)
        plt.ylabel('Speedup from slowest')
        plt.tight_layout()
        newname = '{0}-{1}-{2}.svg'.format(name, host, p)
        plt.savefig(os.path.join(outdir, newname), format='svg')
        print('Created', os.path.join(outdir, newname))
        plt.cla()

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
