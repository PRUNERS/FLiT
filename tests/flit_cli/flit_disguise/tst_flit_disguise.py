#!/usr/bin/env python3
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

'''
Tests FLiT's disguise subcommand as integration tests
'''

import unittest as ut
import tempfile
from io import StringIO
import re

import sys
before_path = sys.path[:]
sys.path.append('../..')
import test_harness as th
sys.path = before_path

NamedTempFile = lambda: tempfile.NamedTemporaryFile(mode='wt', buffering=1)

class FlitTestBase(ut.TestCase):

    def capture_flit(self, args):
        '''
        Runs the flit command-line tool with the given args.  Returns the
        standard output from the flit run as a list of lines.
        '''
        with StringIO() as ostream:
            retval = th.flit.main(args, outstream=ostream)
            lines = ostream.getvalue().splitlines()
        self.assertEqual(retval, 0)
        return lines

    def run_flit(self, args):
        'Runs flit ignoring standard output'
        self.capture_flit(args)

class FlitDisguiseTest(FlitTestBase):

    def setup_flitdir(self, directory):
        self.run_flit(['init', '--directory', directory])

    def disguise_string(self, content, fields=None, mapping=None, undo=False):
        'Runs flit disguise on the content and returns the disguised version'
        with NamedTempFile() as fcontent:
            fcontent.write(content)
            fcontent.flush()
            args = ['disguise', fcontent.name]

            if fields is not None:
                args.extend(['--fields', ','.join(fields)])

            if undo:
                args.append('--undo')

            if mapping is not None:
                with NamedTempFile() as fout:
                    fout.write('disguise,value\n')
                    fout.file.writelines(['"{}","{}"\n'.format(key, value)
                        for key, value in mapping.items()])
                    fout.flush()
                    args.extend(['--disguise-map', fout.name])
                    return self.capture_flit(args)

            return self.capture_flit(args)

    def test_generate_map_default_flit_init(self):
        with th.util.tempdir() as flitdir:
            self.setup_flitdir(flitdir)
            with th.util.pushd(flitdir):
                output = self.capture_flit(['disguise', '--generate'])
                self.assertEqual(output, ['Created disguise.csv'])
                with open('disguise.csv') as disguise_in:
                    disguise_contents = disguise_in.readlines()
        self.assertEqual('disguise,value\n', disguise_contents[0])
        self.assertEqual('objfile-1,Empty.cpp.o\n', disguise_contents[1])
        self.assertEqual('objfile-2,main.cpp.o\n', disguise_contents[2])
        self.assertEqual('filepath-1,main.cpp\n', disguise_contents[3])
        self.assertEqual('filepath-2,tests/Empty.cpp\n', disguise_contents[4])
        self.assertEqual('filename-1,Empty.cpp\n', disguise_contents[5])
        self.assertEqual('test-1,Empty\n', disguise_contents[-1])
        expected_symbols = ['main']
        symbol_lines = [x for x in disguise_contents if x.startswith('symbol')]
        for symbol in expected_symbols:
            self.assertTrue(any(re.match('symbol-\d*,{}\n'.format(symbol), line)
                                for line in disguise_contents))

    def test_disguise_empty_map(self):
        to_disguise = [
            '  Just some file contents.',
            'Nothing to worry about here.',
            ]
        disguised = self.disguise_string('\n'.join(to_disguise), mapping={})
        self.assertEqual(disguised, to_disguise)

    def test_disguise_normal(self):
        disguise_mapping = {
            'disguise-01': 'map',
            'disguise-02': 'hi',
            'disguise-03': 'function(string, int, int)',
            'disguise-04': 'not found',
            }
        to_disguise = [
            'hi there chico',
            'may mapping map file is not so good',
            'has a function called function(string, int, int).',
            ]
        expected_disguised = [
            'disguise-02 there chico',
            'may mapping disguise-01 file is not so good',
            'has a function called disguise-03.',
            ]
        disguised = self.disguise_string(
            '\n'.join(to_disguise), mapping=disguise_mapping)
        self.assertEqual(disguised, expected_disguised)

    def test_disguise_normal_undo(self):
        disguise_mapping = {
            'disguise-01': 'map',
            'disguise-02': 'hi',
            'disguise-03': 'function(string, int, int)',
            'disguise-04': 'not found',
            }
        disguised = [
            'disguise-02 there chico',
            'may mapping disguise-01 file is not so good',
            'has a function called disguise-03.',
            ]
        expected_undisguised = [
            'hi there chico',
            'may mapping map file is not so good',
            'has a function called function(string, int, int).',
            ]

        undisguised = self.disguise_string(
            '\n'.join(disguised), mapping=disguise_mapping, undo=True)
        self.assertEqual(undisguised, expected_undisguised)

    def test_disguise_bad_map_file(self):
        with NamedTempFile() as mapfile:
            mapfile.write('not the correct header\n'
                          'does not matter\n'
                          'what the rest has...\n')
            with self.assertRaises(AssertionError):
                self.run_flit(['disguise', '--disguise-map', mapfile.name])

if __name__ == '__main__':
    sys.exit(th.unittest_main())
