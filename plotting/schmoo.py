#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

import extractor

import numpy as np

# note: the /etc/matplotlibrc (on Ubuntu 16)  file has to be configured to use
# the 'Agg' backend (see the file for details).  This is so it will work
# in the PostgreSql environment

def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)


def plot(x_ticks, y_ticks, z_data, file_name, title, labsize):
    print('Saving figure to', file_name)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    #fig.suptitle(title, fontsize=8)
    X = np.array(z_data)

    ax.imshow(X, cmap=cm.hot, interpolation='nearest')

    numrows, numcols = X.shape
    ax.format_coord = format_coord

    plt.xticks(np.arange(len(x_ticks)), tuple(x_ticks), rotation='vertical')

    plt.yticks(np.arange(len(y_ticks)), tuple(y_ticks), rotation='horizontal')

    ax.tick_params(axis='both', which='major', labelsize=labsize)
    ax.tick_params(axis='both', which='minor', labelsize=labsize)
    # ax.set_xticklabels(xticklabels, fontsize=6)
    # ax.set_xticklabels(xticklabels, fontsize=6)
    # ax.set_yticklabels(yticklabels, fontsize=6)
    #plt.xticks(np.arange(6), ('a', 'b', 'c', 'd', 'e', 'f'))

    plt.tight_layout()

    plt.savefig(file_name)
    #plt.show()

    #pl.plot(x_ticks, y_ticks, z_data, fname)

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        Generates schmoo plots directly from the database.
        The default behavior is to only extract the most recent run.
        ''')
    parser.add_argument('-r', '--run', metavar='N', type=int, default=-1,
                        help='Which run to use.  Default is the latest run.')
    parser.add_argument('-p', '--precision', choices=['f', 'd', 'e'], default='f',
                        help='Which precision to use.  Default is "f" for float')
    parser.add_argument('-c', '--compiler', type=str, default='g++',
                        help='Which compiler to use.  Default is "g++"')
    parser.add_argument('-o', '--output', type=str, default='output.png',
                        help='Where to output resultant plot.  Default is output.png')
    #parser.add_argument('-l', '--list', action='store_true',
    #                    help='List the avilable runs for download')
    args = parser.parse_args(args=arguments)

    db = extractor.connect_to_database()
    
    # TODO: make args.precision be a list of precisions
    # TODO: make args.compiler be a list of compilers
    # TODO: get --list command to work with run, precision, or compiler
    prec_str = " and precision = '{0}'".format(args.precision)
    comp_str = " and compiler = '{0}'".format(args.compiler)

    quer = ("select" +
            " distinct name from tests as t1" +
            " where exists" +
            "  (select 1 from tests" + 
            "   where t1.name = name" +
            "    and t1.precision = precision" +
            "    and t1.score0 != score0" +
            "    and t1.run = run" +
            "    and t1.compiler = compiler" +
            "  )" +
            " and run = {0}".format(args.run) +
            " and precision = '{0}'".format(args.precision) +
            " and compiler = '{0}'".format(args.compiler) +
            " order by name")
    tests = extractor.run_query(db, quer)

    tests_str = ""
    if len(tests) > 0:
        tests_str = " and (name = '"
        for t in tests:
            tests_str += t['name'] + "' or name = '"
        tests_str = tests_str[:-12] + ")"

    quer = ("select" +
            " distinct switches, compiler, precision" +
            " from tests"
            " where" +
            "  run = {0}".format(args.run) +
            "  and precision = '{0}'".format(args.precision) +
            "  and compiler = '{0}'".format(args.compiler) +
            "  " + tests_str +
            " order by compiler, switches")
    x_axis = extractor.run_query(db, quer)
    xa_count = len(x_axis)

    x_ticks = []
    y_ticks = []
    z_data = []

    x_count = 0
    y_count = 0

    ret_val = ""

    for x in x_axis:
        x_ticks.append(x['switches'])
    scores = []
    for t in tests:
        y_ticks.append(t['name'])
        y_count += 1
        del scores[:]
        quers = ("select distinct score0, switches, " +
                 "compiler, optl from " +
                 " tests where run = " + str(args.run) +
                 " and name = '" + t['name'] + "'" + prec_str + comp_str +
                 " and optl = '-O0'" +
                 " order by compiler, optl, switches")
        scores.append(extractor.run_query(db, quers))
        x_axis = extractor.run_query(db, quers)
        quers = ("select distinct score0, switches, " +
                 "compiler, optl from " +
                 " tests where run = " + str(args.run) +
                 " and name = '" + t['name'] + "'" + prec_str + comp_str +
                 " and optl = '-O1'" +
                 " order by compiler, optl, switches")
        scores.append(extractor.run_query(db, quers))
        quers = ("select distinct score0, switches, " +
                 "compiler, optl from " +
                 " tests where run = " + str(args.run) +
                 " and name = '" + t['name'] + "'" + prec_str + comp_str +
                 " and optl = '-O2'" +
                 " order by compiler, optl, switches")
        scores.append(extractor.run_query(db, quers))
        quers = ("select distinct score0, switches, " +
                 "compiler, optl from " +
                 " tests where run = " + str(args.run) +
                 " and name = '" + t['name'] + "'" + prec_str + comp_str +
                 " and optl = '-O3'" +
                 " order by compiler, optl, switches")
        scores.append(extractor.run_query(db, quers))
        eq_classes = []
        del eq_classes[:]
        line_classes = []
        for set in scores:
            color = 0
            eq = {}
            for x in set:
                if not x['score0'] in eq:
                    eq[x['score0']] = color
                    color += 1
            eq_classes.append(eq)
        for x in x_axis:
            quer = ("select distinct score0, optl from tests where name = '" +
                    t['name'] + "' and precision = '" + args.precision +
                    "' and switches = '" + x['switches'] +
                    "' and compiler = '" + x['compiler'] +
                    "' and run = " + str(args.run) + " order by optl")
            score = extractor.run_query(db, quer)
            #assert len(score) == 4, score
            x_count += 1
            test_scores = []
            for de in zip(score, eq_classes):
                test_scores.append(de[1][de[0]['score0']])
            line_classes.append(test_scores)
        z_data.append(line_classes)

    plot(x_ticks, y_ticks, z_data, args.output,
            args.compiler + ' @ precision: ' + args.precision,
            labsize=4)

if __name__ == '__main__':
    main(sys.argv[1:])
