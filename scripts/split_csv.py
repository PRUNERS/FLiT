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
    parser.add_argument('--by', required=True, help='split by column values')
    parser.add_argument('--level-load', type=int, default=None,
                        help='''
                            level-load using the --by column and treat it as a
                            number.  The integer here is how many splits to
                            perform.
                            ''')
    return parser.parse_args(arguments)

def level_load(rows, by, splitcount):
    'Split by level load'
    argmin = lambda arr: min(enumerate(arr), key=lambda x: x[1])
    rows.sort(key=lambda x: -x[by])
    split = [[] for _ in range(splitcount)]
    splitload = [0 for _ in range(splitcount)]
    for row in rows:
        idx, _ = argmin(splitload)
        split[idx].append(row)
        splitload[idx] += row[by]
    print('Split counts and time:')
    for i, rowlist in enumerate(split):
        print('  #{}: {} ({} sec)'.format(i, len(rowlist), splitload[i] / 1.0e9))
    return split

def split_val(rows, by):
    'Split by unique values in the by column'
    split = {val: [x for x in rows if x[by] == val]
             for val in {x[by] for x in rows}}
    print('Split counts:')
    for val, rowlist in sorted(split.items()):
        print('  {}: {}'.format(val, len(rowlist)))
    return [val for _, val in sorted(split.items())]

def main(arguments):
    'Main logic here'
    args = parse_args(arguments)
    base, ext = os.path.splitext(args.csv)
    with open(args.csv, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        rows = list(reader)
    if args.level_load is None:
        split = split_val(rows, args.by)
    else:
        split = level_load(rows, args.by, args.level_load)
    i = 1
    for rowlist in split:
        with open('{}-{:02}{}'.format(base, i, ext), 'w') as csv_out:
            writer = csv.DictWriter(csv_out, rows[0].keys())
            writer.writeheader()
            writer.writerows(rowlist)
        i += 1

if __name__ == '__main__':
    main(sys.argv[1:])
