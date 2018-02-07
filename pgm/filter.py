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
