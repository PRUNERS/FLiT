#!/usr/bin/env python3

import pivot_table
import csv
import sys
import argparse
import itertools

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
    parser.add_argument('-r', '--rows', default='switches',
                        help='''
                            Comma-separated list of table columns to use as the
                            rows and consequently columns.  For example, --rows
                            name,compiler
                            ''')
    parser.add_argument('-f', '--fix', default=None,
                        help='''
                            Comma-separated list of name=value to remain fixed
                            (i.e.  pre-filter).  For example, --fix
                            precision=d,switches=,compiler=g++
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

    table = []  # list of lists, len(row_names) x len(row_names)
    for row_name in row_names:
        new_table_row = []
        table.append(new_table_row)
        row_items = [x for x in inputrows
                     if all(x[name] == value
                            for name, value in zip(row_types, row_name))]
        for col_name in row_names:
            col_items = [x for x in inputrows
                         if all(x[name] == value
                                for name, value in zip(row_types, col_name))]
            new_table_row.append(sum(x['score0'] != y['score0']
                                     for x in row_items for y in col_items
                                     if all(x[name] == y[name]
                                            for name in other_types)
                                ))
        print(new_table_row)
    return (row_names, table)

def main(arguments):
    'Main entry point'
    args = parse_args(arguments)
    filters = pivot_table.split_filters(args.fix)
    inputrows = pivot_table.read_csv(args.csvfile, filters)

    row_types = args.rows.split(',')
    [row_names, table] = generate_compare_matrix(inputrows, row_types)

    if args.stdout:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    try:
        pivot_table.write_table_to_csv(outfile, row_names, row_names, table)
    except:
        if not args.stdout:
            outfile.close()
        raise


if __name__ == '__main__':
    main(sys.argv[1:])
