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

>>> with th.tempdir() as temp_dir:
...     with StringIO() as ostream:
...         _ = th.flit.main(['init', '-C', temp_dir], outstream=ostream)
...         init_out = ostream.getvalue().splitlines()
...     shutil.rmtree(os.path.join(temp_dir, 'tests'))
...     _ = shutil.copytree(os.path.join('data', 'tests'),
...                         os.path.join(temp_dir, 'tests'))
...     with StringIO() as ostream:
...         _ = th.flit.main(['bisect', '-C', temp_dir,
...                           '--precision', 'double',
...                           'g++ -O3', 'BisectTest',
...                           '--biggest', '1'],
...                          outstream=ostream)
...         bisect_out_1 = ostream.getvalue().splitlines()
...     with StringIO() as ostream:
...         _ = th.flit.main(['bisect', '-C', temp_dir,
...                           '--precision', 'double',
...                           'g++ -O3', 'BisectTest',
...                           '--biggest', '2'],
...                          outstream=ostream)
...         bisect_out_2 = ostream.getvalue().splitlines()
...     with open(os.path.join(temp_dir, 'bisect-01', 'bisect.log')) as fin:
...         log_contents = fin.readlines()

Verify the output of flit init
>>> print('\\n'.join(init_out)) # doctest:+ELLIPSIS
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile

>>> print('\\n'.join(bisect_out_1)) # doctest:+ELLIPSIS
Updating ground-truth results - ground-truth.csv - done
Looking for the top 1 different symbol(s) by starting with files
  Created /.../bisect-01/bisect-make-01.mk - compiling and running - score 21.0
  Created /.../bisect-01/bisect-make-02.mk - compiling and running - score 14.0
  Created /.../bisect-01/bisect-make-03.mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-04.mk - compiling and running - score 0.0
  Created /.../bisect-01/bisect-make-05.mk - compiling and running - score 14.0
  Created /.../bisect-01/bisect-make-06.mk - compiling and running - score 10.0
  Created /.../bisect-01/bisect-make-07.mk - compiling and running - score 4.0
    Found differing source file tests/file1.cpp: score 10.0
    Searching for differing symbols in: tests/file1.cpp
      Created ...
      ...
        Found differing symbol on line 92 -- file1_func2_PROBLEM() (score 5.0)
  Created /.../bisect-01/bisect-make-15.mk - compiling and running - score 0.0
  Created /.../bisect-01/bisect-make-16.mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-17.mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-18.mk - compiling and running - score 0.0
    Found differing source file tests/file2.cpp: score 7.0
    Searching for differing symbols in: tests/file2.cpp
      Created ...
      ...
        Found differing symbol on line 91 -- file2_func1_PROBLEM() (score 7.0)
    Found differing source file tests/file3.cpp: score 4.0
found highest variability inducing source files:
  tests/file1.cpp (score 10.0)
  tests/file2.cpp (score 7.0)
  tests/file3.cpp (score 4.0)
1 highest variability symbol from each found source file:
  /.../tests/file1.cpp:92 _Z19file1_func2_PROBLEMv -- file1_func2_PROBLEM() (score 5.0)


>>> print('\\n'.join(bisect_out_1)) # doctest:+ELLIPSIS
Updating ground-truth results - ground-truth.csv - done
Looking for the top 1 different symbol(s) by starting with files
  Created /.../bisect-01/bisect-make-01.mk - compiling and running - score 21.0
  Created /.../bisect-01/bisect-make-02.mk - compiling and running - score 14.0
  Created /.../bisect-01/bisect-make-03.mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-04.mk - compiling and running - score 0.0
  Created /.../bisect-01/bisect-make-05.mk - compiling and running - score 14.0
  Created /.../bisect-01/bisect-make-06.mk - compiling and running - score 10.0
  Created /.../bisect-01/bisect-make-07.mk - compiling and running - score 4.0
    Found differing source file tests/file1.cpp: score 10.0
    Searching for differing symbols in: tests/file1.cpp
      Created ...
      ...
        Found differing symbol on line 92 -- file1_func2_PROBLEM() (score 5.0)
  Created /.../bisect-01/bisect-make-...mk - compiling and running - score 0.0
  Created /.../bisect-01/bisect-make-...mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-...mk - compiling and running - score 7.0
  Created /.../bisect-01/bisect-make-...mk - compiling and running - score 0.0
    Found differing source file tests/file2.cpp: score 7.0
    Searching for differing symbols in: tests/file2.cpp
      Created ...
      ...
        Found differing symbol on line 91 -- file2_func1_PROBLEM() (score 7.0)
    Found differing source file tests/file3.cpp: score 4.0
The found highest variability inducing source files:
  tests/file1.cpp (score 10.0)
  tests/file2.cpp (score 7.0)
  tests/file3.cpp (score 4.0)
The 1 highest variability symbol:
  /.../tests/file2.cpp:91 _Z19file2_func1_PROBLEMv -- file2_func1_PROBLEM() (score 7.0)

>>> print('\\n'.join(bisect_out_2)) # doctest:+ELLIPSIS
Updating ground-truth results - ground-truth.csv - done
Looking for the top 2 different symbol(s) by starting with files
  Created /.../bisect-02/bisect-make-01.mk - compiling and running - score 21.0
  Created /.../bisect-02/bisect-make-02.mk - compiling and running - score 14.0
  Created /.../bisect-02/bisect-make-03.mk - compiling and running - score 7.0
  Created /.../bisect-02/bisect-make-04.mk - compiling and running - score 0.0
  Created /.../bisect-02/bisect-make-05.mk - compiling and running - score 14.0
  Created /.../bisect-02/bisect-make-06.mk - compiling and running - score 10.0
  Created /.../bisect-02/bisect-make-07.mk - compiling and running - score 4.0
    Found differing source file tests/file1.cpp: score 10.0
    Searching for differing symbols in: tests/file1.cpp
      Created ...
      ...
        Found differing symbol on line 92 -- file1_func2_PROBLEM() (score 5.0)
      Created ...
      ...
        Found differing symbol on line 108 -- file1_func4_PROBLEM() (score 3.0)
  Created /.../bisect-02/bisect-make-...mk - compiling and running - score 0.0
  Created /.../bisect-02/bisect-make-...mk - compiling and running - score 7.0
  Created /.../bisect-02/bisect-make-...mk - compiling and running - score 7.0
  Created /.../bisect-02/bisect-make-...mk - compiling and running - score 0.0
    Found differing source file tests/file2.cpp: score 7.0
    Searching for differing symbols in: tests/file2.cpp
      Created ...
      ...
        Found differing symbol on line 91 -- file2_func1_PROBLEM() (score 7.0)
    Found differing source file tests/file3.cpp: score 4.0
The found highest variability inducing source files:
  tests/file1.cpp (score 10.0)
  tests/file2.cpp (score 7.0)
  tests/file3.cpp (score 4.0)
The 2 highest variability symbols:
  /.../tests/file2.cpp:91 _Z19file2_func1_PROBLEMv -- file2_func1_PROBLEM() (score 7.0)
  /.../tests/file1.cpp:92 _Z19file1_func2_PROBLEMv -- file1_func2_PROBLEM() (score 5.0)

TODO: Test the log_contents variable

TODO: This test takes about a whole minute on my laptop.  Let's create mocks
TODO- and stubs to eliminate the dependence on the compiler.  The interaction
TODO- with the real compiler is already tested in tst_bisect.py, so we can
TODO- safely stub it out here.
'''

# Test setup before the docstring is run.
import sys
before_path = sys.path[:]
sys.path.append('..')
import test_harness as th
sys.path = before_path

if __name__ == '__main__':
    from doctest import testmod
    failures, tests = testmod()
    sys.exit(failures)
