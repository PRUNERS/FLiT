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
Tests FLiT's capabilities to run bisect and identify the problem files and
functions

The tests are below using doctest

Let's now make a temporary directory and test that we can successfully compile
and run FLiT bisect

>>> import glob
>>> import os
>>> import shutil
>>> import subprocess as subp
>>> from io import StringIO
>>> import flitutil as util
>>> from common import BisectTestError, flit_init

>>> class BisectTestError(RuntimeError): pass

>>> with th.tempdir() as temp_dir:
...     init_out = flit_init(temp_dir)
...     shutil.rmtree(os.path.join(temp_dir, 'tests'))
...     _ = shutil.copytree(os.path.join('data', 'tests'),
...                         os.path.join(temp_dir, 'tests'))
...     with open(os.path.join(temp_dir, 'custom.mk'), 'a') as mkout:
...         _ = mkout.write('SOURCE += tests/file4.cxx\\n')
...     with StringIO() as ostream:
...         retval = th.flit.main(['bisect', '-C', os.path.relpath(temp_dir),
...                                '--compiler-type', 'gcc',
...                                '--precision', 'double',
...                                'g++ -O3', 'BisectTest'],
...                               outstream=ostream)
...         if retval != 0:
...             raise BisectTestError(
...                 'Could not bisect (1) (retval={0}):\\n'.format(retval) +
...                 ostream.getvalue())
...         bisect_out = ostream.getvalue().splitlines()
...     with open(os.path.join(temp_dir, 'bisect-01', 'bisect.log')) as fin:
...         raw_log = fin.readlines()
...         stripped_log = [line[line.index(' bisect:')+8:].rstrip()
...                         for line in raw_log]
...         log_contents = [line for line in stripped_log if line.strip() != '']
...     troublecxx = util.extract_make_var(
...         'TROUBLE_CXX',
...         os.path.join('bisect-01', 'bisect-make-01.mk'),
...         directory=temp_dir)
...     troublecxx_type = util.extract_make_var(
...         'TROUBLE_CXX_TYPE',
...         os.path.join('bisect-01', 'bisect-make-01.mk'),
...         directory=temp_dir)

Verify the output of flit init
>>> print('\\n'.join(init_out)) # doctest:+ELLIPSIS
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile

Let's see that the ground truth results are updated first
>>> bisect_out[0]
'Updating ground-truth results - ground-truth.csv - done'

Verify that all source files were found and output during the search
>>> sorted([x.split(':')[0].split()[-1] for x in bisect_out
...         if x.startswith('    Found differing source file')]
...       ) # doctest: +NORMALIZE_WHITESPACE
['tests/A.cpp',
 'tests/BisectTest.cpp',
 'tests/file1.cpp',
 'tests/file2.cpp',
 'tests/file3.cpp',
 'tests/file4.cxx']

Verify that the four differing sources were output in the "differing sources:"
section
>>> idx = bisect_out.index('all variability inducing source file(s):')
>>> print('\\n'.join(bisect_out[idx+1:idx+7]))
  tests/BisectTest.cpp (score 50.0)
  tests/file4.cxx (score 30.0)
  tests/file1.cpp (score 10.0)
  tests/file2.cpp (score 7.0)
  tests/file3.cpp (score 4.0)
  tests/A.cpp (score 2.0)
>>> bisect_out[idx+7].startswith('Searching for differing symbols in:')
True

Verify that all four files were searched individually
>>> sorted([x.split()[-1] for x in bisect_out
...         if x.startswith('Searching for differing symbols in:')]
...       ) # doctest: +NORMALIZE_WHITESPACE
['tests/A.cpp',
 'tests/BisectTest.cpp',
 'tests/file1.cpp',
 'tests/file2.cpp',
 'tests/file3.cpp',
 'tests/file4.cxx']

Verify all non-templated functions were identified during the symbol searches
>>> print('\\n'.join(
...     sorted([' '.join(x.split()[4:]) for x in bisect_out
...             if x.startswith('    Found differing symbol on line')])))
line 100 -- file1_func3_PROBLEM() (score 2.0)
line 103 -- file3_func5_PROBLEM() (score 3.0)
line 108 -- file1_func4_PROBLEM() (score 3.0)
line 110 -- file4_all() (score 30.0)
line 90 -- file2_func1_PROBLEM() (score 7.0)
line 92 -- file1_func2_PROBLEM() (score 5.0)
line 92 -- file3_func2_PROBLEM() (score 1.0)
line 95 -- A::fileA_method1_PROBLEM() (score 2.0)
line 96 -- real_problem_test(int, char**) (score 50.0)

Verify the differing symbols section for file1.cpp
>>> idx = bisect_out.index('  All differing symbols in tests/file1.cpp:')
>>> print('\\n'.join(bisect_out[idx+1:idx+4]))
    line 92 -- file1_func2_PROBLEM() (score 5.0)
    line 108 -- file1_func4_PROBLEM() (score 3.0)
    line 100 -- file1_func3_PROBLEM() (score 2.0)
>>> bisect_out[idx+4].startswith(' ')
False

Verify the differing symbols section for file2.cpp
>>> idx = bisect_out.index('  All differing symbols in tests/file2.cpp:')
>>> bisect_out[idx+1]
'    line 90 -- file2_func1_PROBLEM() (score 7.0)'
>>> bisect_out[idx+2].startswith(' ')
False

Verify the differing symbols section for file3.cpp
>>> idx = bisect_out.index('  All differing symbols in tests/file3.cpp:')
>>> print('\\n'.join(bisect_out[idx+1:idx+3]))
    line 103 -- file3_func5_PROBLEM() (score 3.0)
    line 92 -- file3_func2_PROBLEM() (score 1.0)
>>> bisect_out[idx+3].startswith('    ')
False

Verify the bad symbols section for file4.cxx
>>> idx = bisect_out.index('  All differing symbols in tests/file4.cxx:')
>>> print('\\n'.join(sorted(bisect_out[idx+1:idx+2])))
    line 110 -- file4_all() (score 30.0)
>>> bisect_out[idx+2].startswith(' ')
False

Verify the bad symbols section for fileA.cpp
>>> idx = bisect_out.index('  All differing symbols in tests/A.cpp:')
>>> print('\\n'.join(sorted(bisect_out[idx+1:idx+2])))
    line 95 -- A::fileA_method1_PROBLEM() (score 2.0)
>>> bisect_out[idx+2].startswith(' ')
False

Verify the bad symbols section for BisectTest.cpp
>>> idx = bisect_out.index('  All differing symbols in tests/BisectTest.cpp:')
>>> print('\\n'.join(sorted(bisect_out[idx+1:idx+2])))
    line 96 -- real_problem_test(int, char**) (score 50.0)
>>> bisect_out[idx+2].startswith(' ')
False

Test the All differing symbols section of the output
>>> idx = bisect_out.index('All variability inducing symbols:')
>>> print('\\n'.join(bisect_out[idx+1:])) # doctest:+ELLIPSIS
  /.../tests/BisectTest.cpp:96 ... -- real_problem_test(int, char**) (score 50.0)
  /.../tests/file4.cxx:110 ... -- file4_all() (score 30.0)
  /.../tests/file2.cpp:90 ... -- file2_func1_PROBLEM() (score 7.0)
  /.../tests/file1.cpp:92 ... -- file1_func2_PROBLEM() (score 5.0)
  /.../tests/file1.cpp:108 ... -- file1_func4_PROBLEM() (score 3.0)
  /.../tests/file3.cpp:103 ... -- file3_func5_PROBLEM() (score 3.0)
  /.../tests/A.cpp:95 ... -- A::fileA_method1_PROBLEM() (score 2.0)
  /.../tests/file1.cpp:100 ... -- file1_func3_PROBLEM() (score 2.0)
  /.../tests/file3.cpp:92 ... -- file3_func2_PROBLEM() (score 1.0)

Test that the --compiler-type flag value made it into the bisect Makefile
>>> troublecxx
['g++']
>>> troublecxx_type
['gcc']

TODO: test the log_contents variable
'''

# Test setup before the docstring is run.
import sys
before_path = sys.path[:]
sys.path.append('../..')
import test_harness as th
sys.path = before_path

if __name__ == '__main__':
    from doctest import testmod
    failures, tests = testmod()
    sys.exit(failures)
