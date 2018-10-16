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
    return parser.parse_args(arguments)

def main(arguments):
    'Main logic here'
    args = parse_args(arguments)
    base, ext = os.path.splitext(args.csv)
    with open(args.incsv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        rows = list(reader)
    with open(args.filtercsv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        filterrows = list(reader)
    matching = [row for row in rows
                if any(all(row[x] == other[x] for x in args.by)
                       for other in filterrows)]
    with open('{}-filtered{}'.format(base, ext), 'w') as csv_out:
        writer = csv.DictWriter(csv_out, rows[0].keys())
        writer.writeheader()
        writer.writerows(matching)

if __name__ == '__main__':
    main(sys.argv[1:])
