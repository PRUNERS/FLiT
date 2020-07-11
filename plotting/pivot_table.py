#!/usr/bin/env python

# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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
#   https://github.com/PRUNERS/FLiT/blob/main/LICENSE
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

'Generates statistics from a given test results csv file'

import csv
import sys
import argparse
import itertools

def parse_args(arguments):
    'Parses the arguments and returns a namedtuple according to argparse'
    parser = argparse.ArgumentParser(
        description='''
            Generates statistics from a given test results csv file.  The csv
            file should have columns "switches", "precision", "sort", "score0",
            "score0d", "host", "compiler", "name", "score1", "score1d".  This
            basically mimics a pivot table functionality.
            ''',
        )
    parser.add_argument('-c', '--columns', default=None,
                        help='''
                            Comma-separated list of table columns to use as the
                            columns.  If empty, then it will just to totals by
                            row.  For example, --columns precision,sort
                            ''')
    parser.add_argument('-r', '--rows', default='name',
                        help='''
                            Comma-separated list of table columns to use as the
                            rows.  For example, --rows name,compiler
                            ''')
    parser.add_argument('-f', '--fix', default=None,
                        help='''
                            Comma-separated list of name=value to remain fixed
                            (i.e.  pre-filter).  For example, --fix
                            precision=d,switches=,compiler=g++
                            ''')
    parser.add_argument(
        '-F', '--format', choices=['csv', 'latex'], default='csv',
        help='''
            The output file format.  Default is csv.
            ''')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--output', default='output.csv',
                       help='Output file for generated table counts')
    group.add_argument('-', '--stdout', action='store_true',
                       help='''
                           Write to stdout instead of to a file.  Conflicts
                           with --output.
                           ''')
    parser.add_argument('csvfile')
    return parser.parse_args(args=arguments)


def split_filters(filter_string):
    '''
    Returns a list of 2-tuples of (name, value) in the order given.  The
    filter_string is of the format key1=value1,key2=value2,key3=value3 for as
    many key-value pairs you want.

    You may enclose a value in single or double quotes, but you cannot have a
    comma in a value field since that separates the pairs.
    '''
    if filter_string is None:
        return []

    filters = []
    for fixed_filter in filter_string.split(','):
        filter_split = fixed_filter.split('=')
        assert len(filter_split) == 2, \
                fixed_filter + ' is not a valid --fix parameter'
        filters.append((filter_split[0], filter_split[1].strip('"\'')))
    return filters


def read_csv(filename, filters):
    '''
    Read in the csv applying the filters of only keeping rows that match the
    column -> value pairs in all filters
    '''
    inputrows = []
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        inputrows = [x for x in reader
                     if all(x[name] == value for name, value in filters)]
    return inputrows


def generate_table(inputrows, col_types, row_types):
    '''
    Creates the table complete with column names and row names

    Returns a tuple of (col_names, row_names, table)
    '''
    row_sets = [set(x[name] for x in inputrows) for name in row_types]
    col_sets = [set(x[name] for x in inputrows) for name in col_types]
    row_names = [x for x in itertools.product(*[sorted(x) for x in row_sets])]
    col_names = [x for x in itertools.product(*[sorted(x) for x in col_sets])]

    assert len(row_names) > 0, \
            'You must select at least one row type using --rows'

    # Do some caching to speed this up
    vals2key = lambda vals: '_'.join(vals)
    row2rowkey = lambda row: vals2key(row[x] for x in row_types)
    row2colkey = lambda row: vals2key(row[x] for x in col_types)
    value_map = {}
    for row in inputrows:
        rowkey = row2rowkey(row)
        colkey = row2colkey(row)
        if rowkey not in value_map:
            value_map[rowkey] = {}
        if colkey not in value_map[rowkey]:
            value_map[rowkey][colkey] = set()
        value_map[rowkey][colkey].add(row['score0'])

    table = []
    for row_idx, row_name in enumerate(row_names):
        new_table_row = []
        table.append(new_table_row)
        row_dict = value_map[vals2key(row_name)]
        if len(col_names) > 0:
            for col_name in col_names:
                try:
                    new_table_row.append(len(row_dict[vals2key(col_name)]))
                except KeyError:
                    new_table_row.append(0)
        else:
            try:
                new_table_row.append(len(row_dict['']))
            except KeyError:
                new_table_row.append(0)

    if len(col_names[0]) == 0:
        col_names = [('total',)]

    return (col_names, row_names, table)


def write_table_to_csv(outfile, col_names, row_names, table):
    '''
    Writes the table complete with column names, row names, and table contents
    out to the specified output.
    '''
    writer = csv.writer(outfile)

    fill_empty = lambda x: '<empty>' if x == '' else x

    # Write the first row
    if len(col_names[0]) > 1:
        writer.writerow([r'r\c'] + [str(x) for x in col_names])
    else:
        writer.writerow([r'r\c'] + [fill_empty(x[0]) for x in col_names])

    # Write each row
    for name, row in zip(row_names, table):
        if len(name) == 1:
            writer.writerow([fill_empty(name[0])] + row)
        else:
            writer.writerow([str(name)] + row)

def write_table_to_latex(outfile, col_names, row_names, table, classes=None):
    '''
    Writes the table complete with column names, row names, and table contents
    out to a Latex source file.  The classes are the equivalence classes if
    doing that.  A separate table will be generated for the classes.
    '''
    fill_empty = lambda x: r'$<$empty$>$' if x == '' else x

    if len(col_names[0]) > 1:
        col_names = [str(x) for x in col_names]
    else:
        col_names = [fill_empty(x[0]) for x in col_names]
    col_names = [x.strip().replace('_', '\\_') for x in col_names]

    if len(row_names[0]) > 1:
        row_names = [str(x) for x in row_names]
    else:
        row_names = [fill_empty(x[0]) for x in row_names]
    row_names = [x.strip().replace('_', '\\_') for x in row_names]

    if classes is not None:
        classes = tuple(tuple(fill_empty(y) for y in x.split(', '))
                        for x in classes)

    max_value = max([max(x) for x in table])
    bound = lambda lower, val, upper: min(upper, max(lower, val))
    def colorstring(value):
        'Creates a "\\color[rgb]{r,g,b}" string for the hard-coded colormap'
        percent = value / max_value
        red = bound(0.5, 0.5 + percent * 5 / 3, 1.0)
        green = bound(0.5, 0.125 + percent * 5 / 4, 1.0)
        blue = bound(0.5, -2/3 + percent * 5 / 3, 1.0)
        string = r'\cellcolor[rgb]{' + \
                 '{0}, {1}, {2}'.format(red, green, blue) + \
                 '}'
        return string

    outfile.write(r'''
\documentclass[border=10pt]{standalone}
\renewcommand*{\familydefault}{\sfdefault}
\usepackage{sfmath}
\usepackage{graphicx}
\usepackage{colortbl}
\usepackage{xcolor}

\begin{document}

''')
    outfile.write(r'\begin{tabular}{r' + '|c' * len(col_names) + '}\n')
    outfile.write('  &\n')
    outfile.write('  \\rotatebox{90}{' +
                  '} &\n  \\rotatebox{90}{'.join(col_names) + '}\n')
    outfile.write('  \\\\\n')
    outfile.write('  \\hline\n')
    rows = []
    for i in range(len(row_names)):
        row_string = '  ' + row_names[i] + ' &\n  '
        row_string += ' &\n  '.join(
            [colorstring(x) + ' ' + str(x) for x in table[i]]
            )
        #for elem in table[i][:-1]:
        #    row_string += '  ' + str(elem) + ' &\n'
        #row_string += '  ' + str(table[i][-1]) + '\n'
        rows.append(row_string)
    outfile.write('  \\\\\n  \\hline\n'.join(rows))

    outfile.write('\n\\end{tabular}\n')

    if classes is not None:
        write_class_table_to_latex(outfile, row_names, classes)

    outfile.write('\n\\end{document}\n')

def write_class_table_to_latex(outfile, row_names, classes):
    '''
    This only creates a class table section of a LaTeX document, not a complete
    LaTeX file like write_table_to_latex() does.  This will take the list of
    classes and create a 2-column table with no colors.
    '''
    #outfile.write('\\begin{tabular}{c|l}\n')
    outfile.write('\\begin{tabular}{c|p{9cm}}\n')
    row_strings = []
    for i in range(len(row_names)):
        row_strings.append('{0} & {1}'.format(row_names[i], ', '.join(classes[i])))
        row_strings[-1] = row_strings[-1].replace('_', '\\_')
    outfile.write('  ' + ' \\\\ \\hline\n  '.join(row_strings))
    outfile.write('\n\\end{tabular}\n')


def main(arguments):
    'Main entry point'
    args = parse_args(arguments)
    filters = split_filters(args.fix)
    inputrows = read_csv(args.csvfile, filters)

    if args.columns is not None:
        col_types = args.columns.split(',')
    else:
        col_types = []
    row_types = args.rows.split(',')

    [col_names, row_names, table] = \
            generate_table(inputrows, col_types, row_types)

    if args.stdout:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    try:
        if args.format == 'csv':
            write_table_to_csv(outfile, col_names, row_names, table)
        else:
            write_table_to_latex(outfile, col_names, row_names, table)
    except:
        if not args.stdout:
            outfile.close()
        raise


if __name__ == '__main__':
    main(sys.argv[1:])

