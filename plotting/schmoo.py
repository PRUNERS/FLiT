#!/usr/bin/env python3

import argparse
import sys

import extractor

import numpy as np

from collections import OrderedDict

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
    parser.add_argument('-p', '--precision', default='f',
                        help='Which precision to use.  Default is "f" for float.'
                             '  Note: if you want to use more than one, list them'
                             '  separated by a comma')
    parser.add_argument('-c', '--compiler', default='g++',
                        help='Which compiler to use.  Default is "g++".'
                             '  Note: if you want to use more than one, list them'
                             '  separated by a comma')
    parser.add_argument('-H', '--host',
                        help='Which host to use.  Must specify a value here.'
                             '  Note: if you want to use more than one, list them'
                             '  separated by a comma')
    parser.add_argument('-O', '--optl', default='-O0',
                        help='Which optimization levels to include.  Default is -O0.'
                             '  Note: if you want to use more than one, list them'
                             '  separated by a comma')
    parser.add_argument('-o', '--output', default='output.png',
                        help='Where to output resultant plot.  Default is output.png')
    #parser.add_argument('-l', '--list', action='store_true',
    #                    help='List the avilable runs for download')
    args = parser.parse_args(args=arguments)
    
    hosts = args.host.split(',')
    precisions = args.precision.split(',')
    compilers = args.compiler.split(',')
    optimization_levels = args.optl.split(',')
    
    def print_list(name, values):
        print(name + ':')
        if len(values) > 0:
            print('  ' + '\n  '.join(values))
    print('run: ', args.run)
    print_list('hosts', hosts)
    print_list('precisions', precisions)
    print_list('compilers', compilers)
    print_list('optimization levels', optimization_levels)
    
    def gen_query_sub(field_name, values):
        '''
        Creates a portion of a query for one of the many values to be matched
        '''
        sub = ''
        if len(values) > 0:
            sub = ' and (' + field_name + " = '"
            sub += "' or compiler = '".join(values)
            sub += "')"
        return sub

    host_str = gen_query_sub('host', hosts)
    prec_str = gen_query_sub('precision', precisions)
    comp_str = gen_query_sub('compiler', compilers)
    optl_str = gen_query_sub('optl', optimization_levels)
    
        

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
            "  and run = {0}".format(args.run) +
            "  " + prec_str +
            "  " + optl_str +
            "  " + comp_str +
            "  " + host_str +
            " order by name")
    tests = extractor.run_query(db, quer)
    test_str = gen_query_sub('name', [x['name'] for x in tests])

    quer = ("select" +
            " distinct" +
            "  switches," +
            "  compiler," +
            "  optl," +
            "  precision," +
            "  host" +
            " from tests"
            " where" +
            "  run = {0}".format(args.run) +
            "  " + prec_str +
            "  " + optl_str +
            "  " + comp_str +
            "  " + host_str +
            "  " + test_str +
            " UNION " +
            "select" +
            " distinct" +
            "  switches," +
            "  compiler," +
            "  optl," +
            "  precision," +
            "  host" +
            " from tests where" +
            "  run = {0}".format(args.run) +
            "  " + prec_str +
            "  " + comp_str +
            "  " + host_str +
            "  " + test_str +
            "  and optl = '-O0'" +
            "  and switches = ''" +
            " order by compiler, optl, switches")
    x_axis = extractor.run_query(db, quer)
    xa_count = len(x_axis)

    quer = ("select" +
            " distinct" +
            "  name" +
            " from tests" +
            " where" +
            "  run = {0}".format(args.run) +
            "  " + prec_str +
            "  " + test_str +
            "  " + comp_str +
            " order by name")
    y_axis = extractor.run_query(db, quer)
    ya_count = len(y_axis)

    x_ticks = []
    y_ticks = []
    z_data = []

    x_count = 0
    y_count = 0

    for x in x_axis:
        x_ticks.append(x['switches'] + ' ' + x['optl'])
    for t in y_axis:
        y_ticks.append(t['name'])
        y_count += 1
        quers = ("select" +
                 " distinct" +
                 "  score0," +
                 "  switches," +
                 "  compiler," +
                 "  optl," +
                 "  host" +
                 " from tests where" +
                 "  run = {0}".format(args.run) +
                 "  and name = '{0}'".format(t['name']) +
                 "  " + prec_str +
                 "  " + comp_str +
                 "  " + host_str +
                 "  and optl = '-O0'" +
                 "  and switches = ''" +
                 " UNION " +
                 "select" +
                 " distinct" +
                 "  score0," +
                 "  switches," +
                 "  compiler," +
                 "  optl," +
                 "  host" +
                 " from tests where" +
                 "  run = {0}".format(args.run) +
                 "  and name = '{0}'".format(t['name']) +
                 "  " + prec_str +
                 "  " + comp_str +
                 "  " + optl_str +
                 "  " + host_str +
                 " order by compiler, optl, switches")
        scores = extractor.run_query(db, quers)
        scores = [x['score0'] for x in scores]
        eq_classes = {}
        line_classes = []
        eq_classes = OrderedDict(zip(scores, scores))
        color = 0
        for key in eq_classes:
            eq_classes[key] = color
            color += 1

        for x in x_axis:
            quer = ("select" +
                    " distinct" +
                    "  score0" +
                    " from tests where" +
                    "  name = '{0}'".format(t['name']) +
                    "  and precision = '{0}'".format(x['precision']) +
                    "  and switches = '{0}'".format(x['switches']) +
                    "  and optl = '{0}'".format(x['optl']) +
                    "  and run = {0}".format(args.run) +
                    "  and host = '{0}'".format(x['host']))
            score = extractor.run_query(db, quer)
            x_count += 1
            line_classes.append(eq_classes[score[0]['score0']])
        z_data.append(line_classes)

    plot(x_ticks, y_ticks, z_data, args.output,
            args.compiler + ' @ precision: ' + args.precision,
            labsize=4)

if __name__ == '__main__':
    main(sys.argv[1:])
