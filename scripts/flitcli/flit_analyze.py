# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   John Jacobson (john.jacobson@utah.edu)
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

'Implements the analyze subcommand'

import argparse
import os
import re
import subprocess as subp
import sys

import flitconfig as conf
import flitutil as util

brief_description = 'Aggregates and analyzes log files.'

def populate_parser(parser=None):
    'Populate or create an ArgumentParser'
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.description = '''
            Aggregates log files generated from --logging flag into one
            log file. Has utilities for getting log data in sqlite3 form
            and for plotting timing of log events.
            '''
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')

    parser.add_argument('-l', '--logs', default=False, action='store_true',
                        help='Enable logging of FLiT events.')

    return parser


def main(arguments, prog=None):
    'Main logic here'
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    # TODO: plotting, for each event need different test cases to compare
    
    log_dir = os.path.join(args.directory, 'event_logs')
    event_times = {}

    with util.pushd(log_dir):
        # Read all log files into one final.log
        # If this is too slow, may make more sense to process files in chunks
        fin_list = glob.glob('*.log')
        lines = fileinput.input(fin_list)
        lines.sort(key=lambda k: json.loads(k)['time'])
        with open('final.log', 'w') as fout:
            fout.writelines(lines)
       
        # Now process data read from all logs while it is in memory.
        events = [json.loads(line) for line in lines]
        for event in events:
            name = event['name']
            if name not in event_times:
                event_times[name] = {'start_total': 0, 'stop_total': 0,
                                     'start_count': 0, 'stop_count': 0}
            if event['start_stop'] == 'start':
                event_times[name]['start_total'] += event['time']
                event_times[name]['start_count'] += 1
            else:
                event_times[name]['stop_total'] += event['time']
                event_times[name]['stop_count'] += 1
       
        # Should have equal starts and stops for each event
        for key, val in event_times:
            assert val['start_count'] == val['stop_count'], \
                   'Unequal start/stop count for event: ' + key
            assert val['start_total'] < val['stop_total'], \
                   'Start time should be less than stop time for event: ' + key

        # Plot total time for each log event.
        # util.logplot(event_times)

        # Temp for now, just print the results to console...
        for key, val in event_times:
            print('Event: {}, Elapsed: {}, Dict: {}'.format(key, 
                  val['stop_total']-val['start_total'], json.dumps(val)))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
