#!/usr/bin/env python

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
Executable script for generating comparison matrices.

Call with --help for more information.
'''

import pivot_table

import argparse
import itertools
import sys

def parse_args(arguments):
    'Parses the arguments and returns a namedtuple according to argparse'
    parser = argparse.ArgumentParser(
        description='''
            Generates a comparison matrix between fields of the same table
            column.  The numbers are keeping everything else constant, how many
            of the score0 entries are different between the two.  The csv
            file should have columns "switches", "precision", "sort", "score0",
            "score0d", "host", "compiler", "name", "score1", "score1d".  This
            basically mimics a pivot table functionality.
            ''',
        )
    parser.add_argument(
        '-r', '--rows', default='switches',
        help='''
            Comma-separated list of table columns to use as the
            rows and consequently columns.  By default this is
            set to "switches".  For example, --rows
            name,compiler
            ''')
    parser.add_argument(
        '-f', '--fix', default=None,
        help='''
            Comma-separated list of name=value to remain fixed
            (i.e.  pre-filter).  By default, nothing is fixed.
            For example, --fix
            precision=d,switches=,compiler=g++
            ''')
    parser.add_argument(
        '-F', '--format', choices=['csv', 'latex'], default='csv',
        help='''
            The output file format.  Default is csv.
            ''')
    parser.add_argument(
        '-c', '--classes', action='store_true',
        help='''
            Changes the type of compare matrix to generate to one of
            equivalence classes.  The raw type, which is the default, will
            have all values from the chosen rows.  The classes type will
            have only equivalence classes.  If the output format is csv,
            then these equivalence classes will be output to a file called
            classes.csv (unless outputting to stdout in which case this
            will be output after to stderr, see --stdout).  If the output
            format is latex, then these equivalence classes will be output
            to a table within the same output file.
            ''')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-o', '--output', default='output.csv',
        help='''
            Output file for generated table counts.  This is the
            default behavior with a default output file of
            output.csv.
            ''')
    group.add_argument(
        '-', '--stdout', action='store_true',
        help='''
            Write to stdout instead of to a file.  Conflicts with --output.
            Also causes --classes to output to stderr.
            ''')
    parser.add_argument('csvfile')
    return parser.parse_args(args=arguments)


def generate_compare_matrix(inputrows, row_types):
    '''
    Generates a table of comparisons of the row_types vs row_types.  It
    basically does a count of the number of differences when all other things
    are considered equal.

    Returns a tuple of (row_names, table).
    '''
    total_types = [
        'switches',
        'precision',
        'sort',
        'host',
        'compiler',
        'name',
        ]
    other_types = [x for x in total_types if x not in row_types]
    row_sets = [set(x[name] for x in inputrows) for name in row_types]
    row_names = [x for x in itertools.product(*[sorted(x) for x in row_sets])]
    assert len(row_names) > 0, \
            'You must select at least one row type using --rows'

    # Do some caching to speed this up
    vals2key = lambda vals: '_'.join(vals)
    row2key = lambda row: vals2key(row[x] for x in row_types)
    row2otherkey = lambda row: vals2key(row[x] for x in other_types)
    row_map = {}
    for row in inputrows:
        key = row2key(row)
        otherkey = row2otherkey(row)
        if key not in row_map:
            row_map[key] = {}
        if otherkey not in row_map[key]:
            row_map[key][otherkey] = []
        row_map[key][otherkey].append(row['score0'])

    # list of lists, len(row_names) x len(row_names)
    # initialize the raw_table as empty
    raw_table = [[None] * len(row_names) for x in range(len(row_names))]
    for idx1, row_name in enumerate(row_names):
        row_dict = row_map[vals2key(row_name)]
        for idx2, col_name in enumerate(row_names):
            col_dict = row_map[vals2key(col_name)]
            raw_table[idx1][idx2] = set([
                key
                for key, value in row_dict.items()
                if key in col_dict
                and col_dict[key] != value
                ])
    return (row_names, raw_table)


def generate_classes_matrix(row_names, raw_table):
    '''
    Returns a smaller table with only equivalence classes.  The list of classes
    is also returned.  Each element in the classes list is a comma-separated
    string of row names in that equivalence class.

    Returns the tuple (table, classes)
    '''
    # classrow is a tuple(tuple())
    # two rows are of the same class if they have an equal classrow
    # A row of the table is a list(set())
    row2classrow = lambda row: tuple(tuple(sorted(x)) for x in row)
    classrow_list = []
    kept_idxs = []
    classes = []
    for idx, row in enumerate(raw_table):
        classrow = row2classrow(row)
        if len(row_names[idx]) == 1:
            row_name = row_names[idx][0]
        else:
            row_name = str(row_names[idx])
        if classrow not in classrow_list:
            kept_idxs.append(idx)
            classrow_list.append(classrow)
            classes.append(row_name)
        else:
            class_idx = classrow_list.index(classrow)
            classes[class_idx] += ', ' + row_name
    table = count_raw_table(classrow_list)
    table = [[col for idx, col in enumerate(row) if idx in kept_idxs]
             for row in table]
    return (table, classes)


def count_raw_table(raw_table):
    'Counts the entries in the raw table creating a table only of counts'
    return [[len(x) for x in row] for row in raw_table]


def main(arguments):
    'Main entry point'
    args = parse_args(arguments)
    filters = pivot_table.split_filters(args.fix)
    inputrows = pivot_table.read_csv(args.csvfile, filters)

    row_types = args.rows.split(',')
    [raw_row_names, raw_table] = generate_compare_matrix(inputrows, row_types)

    # Convert the raw_table into a table based on the type requested
    classes = None
    if args.classes:
        table, classes = generate_classes_matrix(raw_row_names, raw_table)
        row_names = ['class-{0}'.format(x + 1) for x in range(len(classes))]
    else:
        table = count_raw_table(raw_table)
        row_names = raw_row_names

    if args.stdout:
        outfile = sys.stdout
        classfile = sys.stderr
    else:
        outfile = open(args.output, 'w')
        if args.classes:
            classfile = open('classes.csv', 'w')

    try:
        if args.format == 'csv':
            pivot_table.write_table_to_csv(outfile, row_names, row_names, table)
            if args.classes and classes is not None:
                pivot_table.write_table_to_csv(classfile, ['class'], row_names,
                                               [[x] for x in classes])
        elif args.format == 'latex':
            pivot_table.write_table_to_latex(outfile, row_names, row_names,
                                             table, classes)

    finally:
        if not args.stdout:
            outfile.close()


if __name__ == '__main__':
    main(sys.argv[1:])
