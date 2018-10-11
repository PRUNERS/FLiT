#!/usr/bin/env python3
'Split csv files into multiple files'

import argparse
import csv
import os
import sys

def parse_args(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--by', required=True)
    return parser.parse_args(arguments)

def main(arguments):
    'Main logic here'
    args = parse_args(arguments)
    base, ext = os.path.splitext(args.csv)
    with open(args.csv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        rows = list(reader)
    split = {val: [x for x in rows if x[args.by] == val]
             for val in set([x[args.by] for x in rows])}
    i = 1
    for key, rowlist in sorted(split.items()):
        with open('{}-{:02}{}'.format(base, i, ext), 'w') as csv_out:
            writer = csv.DictWriter(csv_out, rows[0].keys())
            writer.writeheader()
            writer.writerows(rowlist)
        i += 1

if __name__ == '__main__':
    main(sys.argv[1:])
