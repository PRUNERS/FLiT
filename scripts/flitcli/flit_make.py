# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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

'Implements the make subcommand'

import argparse
import glob
import multiprocessing
import subprocess
import sys

import flitargformatter
import flit_import

brief_description = 'Runs the make and adds to the database'

def populate_parser(parser=None):
    'Populate or create an ArgumentParser'
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.formatter_class = flitargformatter.DefaultsParaSpaciousHelpFormatter
    parser.description = '''
        This command runs the full set of tests and adds the results
        to the configured database.
        '''
    processors = multiprocessing.cpu_count()
    parser.add_argument('-j', '--jobs', type=int, default=processors,
                        help='''
                            The number of parallel jobs to use for the call to
                            GNU make when performing the compilation.  Note,
                            this is not used when executing the tests.  This
                            is because in order to get accurate timing data,
                            one cannot in general run multiple versions of the
                            same code in parallel.
                            ''')
    parser.add_argument('--exec-jobs', type=int, default=1,
                        help='''
                            The number of parallel jobs to use for the call to
                            GNU make when performing the test executtion after
                            the full compilation has finished.  The default is
                            to only run one test at a time in order to allow
                            them to not conflict and to generate accurate
                            timing measurements.
                            ''')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress the Makefile output')
    parser.add_argument('--make-args',
                        help='Arguments passed to GNU Make, separated by commas')
    parser.add_argument('-l', '--label', default='Automatically generated label',
                        help='''
                            The label to attach to the run.  Only applicable
                            when creating a new run.  This argument is ignored
                            if --append is specified.  The default label is
                            ''')
    return parser

def main(arguments, prog=None):
    'Main logic here'
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    check_call_kwargs = dict()
    if args.quiet:
        check_call_kwargs['stdout'] = subprocess.DEVNULL
        #check_call_kwargs['stderr'] = subprocess.DEVNULL
    make_args = []
    if args.make_args is not None:
        make_args = args.make_args.split(',')

    # TODO: can we make a progress bar here?
    print('Calling GNU Make for the runbuild')
    subprocess.check_call([
        'make',
        'runbuild',
        '-j{0}'.format(args.jobs),
        ] + make_args, **check_call_kwargs)
    print('Calling GNU Make to execute the tests')
    subprocess.check_call([
        'make',
        'run',
        '-j{0}'.format(args.exec_jobs),
        ] + make_args, **check_call_kwargs)
    print('Importing into the database')
    # TODO: find a way to not import over again if called multiple times
    status = flit_import.main(['--label', args.label] +
                              glob.glob('results/*_out.csv'))
    if status != 0:
        return status

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
