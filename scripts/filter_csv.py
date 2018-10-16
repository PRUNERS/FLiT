#!/usr/bin/env python3
'Filter csv file using another csv file'

import argparse
import csv
import os
import sys

def parse_args(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser()
    parser.add_argument('incsv')
    parser.add_argument('filtercsv')
    parser.add_argument('--by', required=True, action='append',
                        help='''
                            filter by this column.  Can specify more than once.
                            ''')
    parser.add_argument('-e', '--exclude', action='store_true',
                        help='''
                            Invert the selection.  Instead of keeping those
                            that match, throw away those that match.
                            ''')
    parser.add_argument('-o', '--output', default=None,
                        help='''
                            output to this file instead of the default of
                            <base>-filtered.csv
                            ''')
    args = parser.parse_args(arguments)
    if args.output is None:
        base, ext = os.path.splitext(args.incsv)
        args.output = '{}-filtered{}'.format(base, ext)
    return args

def main(arguments):
    'Main logic here'
    args = parse_args(arguments)
    with open(args.incsv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        rows = list(reader)
    with open(args.filtercsv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        filterrows = list(reader)
    matching = [row for row in rows
                if any(all(row[x] == other[x] for x in args.by)
                       for other in filterrows)]
    tokeep = matching
    if args.exclude:
        tokeep = [row for row in rows if row not in matching]
    with open(args.output, 'w') as csv_out:
        writer = csv.DictWriter(csv_out, rows[0].keys())
        writer.writeheader()
        writer.writerows(tokeep)

if __name__ == '__main__':
    main(sys.argv[1:])
