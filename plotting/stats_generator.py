#!/usr/bin/env python3

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

'Generates statistics from a given test results csv file'

import csv
import sys
import argparse

def main(arguments):
    'Main entry point'
    # Parse arguments
    # TODO: rework this since FLiT has been updated since the creation.
    # TODO: finish generating important statistics
    parser = argparse.ArgumentParser(
        description='''
            Generates statistics from a given test results csv file.  The csv
            file should have columns "switches", "precision", "sort", "score0",
            "score0d", "host", "compiler", "name", "score1", "score1d"
            ''',
        )
    parser.add_argument('csvfile')
    args = parser.parse_args(args=arguments)

    # Read the csvfile into memory
    rows = []
    with open(args.csvfile, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = [x for x in reader]

    # Generate statistics
    names = set(x['name'] for x in rows)
    groups = {}
    scores = {}
    for name in names:
        groups[name] = [x for x in rows if x['name'] == name]
        scores[name] = set(x['score0'] for x in groups[name])
        print(name, len(scores[name]))
    print()



    total_switches = set(x['switches'] for x in rows)
    for switch in total_switches:
        switch_counts[switch] = 0
    for name in names:
        switches = set(x['switches'] for x in groups[name])
        ## TODO: Find a base score to compare against...
        ## TODO-   Without that, we don't have anything to plot on the switches front
        ## TODO-   Maybe we could have the base be
        ## TODO-   - the most common score.
        ## TODO-   - the empty switch for gcc
        ## TODO-   - something else
        #for switch in switches:
        #    switch_counts[switch] += len(set(x['score0'] for x in

if __name__ == '__main__':
    main(sys.argv[1:])

