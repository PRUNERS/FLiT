#!/usr/bin/env python

import csv
import sys
import argparse

class Run(object):
    def __init__(self, csv_file):
        self.csv = csv_file
        self.fields_to_exclude = [
                'score0d',
                'score1',
                'score1d',
                'sort',
                'run',
                'host',
                'file',
                'index',
                ]
        self.field_to_diff = 'score0'
        self.compare_against = {
                'optl': '-O0',
                'switches': '',
                }

        self.rows = []
        self.fields = []
        self.__read_csv()

        self.other_fields = set(self.fields).difference(
                set(self.compare_against.keys() + [self.field_to_diff]))

        self.distinct_vals = {}
        for key in self.fields:
            self.distinct_vals[key] = sorted(set(x[key] for x in self.rows))

        self.compare_rows = {}
        self.__find_compare_rows()

        self.filtered = []
        self.__filter()

    def __read_csv(self):
        with open(self.csv, 'r') as fin:
            reader = csv.DictReader(fin)
            self.rows = [x for x in reader]
            self.fields = [x for x in sorted(self.rows[0].keys())
                    if x not in self.fields_to_exclude]

    def __row_to_other_key(self, row):
        vals = [row[k] for k in self.other_fields]
        return tuple(sorted(vals))

    def __find_compare_rows(self):
        for row in self.rows:
            if all(row[k] == v for k, v in self.compare_against.iteritems()):
                key = self.__row_to_other_key(row)
                self.compare_rows[key] = row

    def __filter(self):
        for row in self.rows:
            newrow = {}
            for key in self.fields:
                if key == self.field_to_diff:
                    other_key = self.__row_to_other_key(row)
                    gt_val = self.compare_rows[other_key][self.field_to_diff]
                    newrow[key] = 1 if gt_val == row[self.field_to_diff] else 0
                else:
                    newrow[key] = self.distinct_vals[key].index(row[key])
            self.filtered.append(newrow)

    def write_distinct_vals(self, outfile):
        with open(outfile, 'w') as optout:
            writer = csv.writer(optout, lineterminator='\n')
            for name in self.distinct_vals.keys():
                writer.writerow(['num', name])
                i = 0
                for val in self.distinct_vals[name]:
                    writer.writerow([i, val])
                    i += 1
                optout.write('\n')

    def write_filtered(self, outcsv):
        with open(outcsv, 'w') as fout:
            writer = csv.DictWriter(fout, self.fields, lineterminator='\n')
            writer.writerow(dict(zip(self.fields,
                sorted(["'" + x + "'" for x in self.fields]))))
            writer.writerows(self.filtered)

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(
            description='''
                Converts incsv into a format
                that can be used by the R
                code and saves it as outfolder.
                ''')
    parser.add_argument('incsv')
    parser.add_argument('outcsv')
    parser.add_argument('-o', '--out-opt-file')
    args = parser.parse_args(args=arguments)

    run = Run(args.incsv)

    if args.out_opt_file is not None:
        run.write_distinct_vals(args.out_opt_file)

    run.write_filtered(args.outcsv)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
