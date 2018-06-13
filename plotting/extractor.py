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

'Extracts test run data into csv files'

from __future__ import print_function

import argparse
import csv
import sys

try:
    import pg
    using_pg = True
except ImportError:
    import psycopg2
    import psycopg2.extras
    using_pg = False

def connect_to_database():
    'Returns a database handle for the framework being used'
    dbname = 'flit'
    host = 'localhost'
    user = 'flit'
    passwd = 'flit123'
    if using_pg:
        return pg.DB(dbname=dbname, host=host, user=user, passwd=passwd)
    else:
        conn_string = "dbname='{dbname}' user='{user}' host='{host}' password='{passwd}'" \
                .format(dbname=dbname, host=host, user=user, passwd=passwd)
        conn = psycopg2.connect(conn_string)
        return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

def run_query(handle, query, *args, **kwargs):
    'Runs the query and returns the result as a list of dictionaries'
    if using_pg:
        return handle.query(query, *args, **kwargs).dictresults()
    else:
        handle.execute(query, *args, **kwargs)
        return [dict(x) for x in handle.fetchall()]

def query_results_to_file(filename, query):
    'Writes results from a PyGresQL query object to a csv file.'
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, sorted(query[0].keys()))
        writer.writeheader()
        writer.writerows(query)

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        Extracts test run data into csv files. Saves runs to run-N.csv.
        The default behavior is to only extract the most recent run.
        ''')
    parser.add_argument('runs', metavar='N', type=int, nargs='*',
                        help='A run to extract to csv.  Saved as run-N.csv')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List the avilable runs for download')
    args = parser.parse_args(args=arguments)

    db = connect_to_database()
    runs = args.runs
    if len(runs) == 0:
        query_runs = run_query(db, 'select index from runs;')
        idx = query_runs[-1]['index']
        runs.append(idx)
        if args.list:
            print('\n'.join([str(x['index']) for x in query_runs]))
            return 0

    for run in runs:
        filename = 'run-{0:02}.csv'.format(run)
        sys.stdout.write('writing ' + filename + ' ...')
        sys.stdout.flush()
        query_tests = run_query(db, 'select * from tests where run = \'{0}\';'.format(run))
        query_results_to_file(filename, query_tests)
        sys.stdout.write(' done\n')
        sys.stdout.flush()
        #print(query_tests)
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])

