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

>>> with th.tempdir() as temp_dir:
...     _ = th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     print()
...     shutil.rmtree(os.path.join(temp_dir, 'tests'))
...     _ = shutil.copytree(os.path.join('data', 'tests'),
...                         os.path.join(temp_dir, 'tests'))
...     _ = th.flit.main(['bisect', '-C', temp_dir, '--precision', 'double',
...                       'g++ -O3', 'BisectTest']) # doctest:+ELLIPSIS
...     with open(os.path.join(temp_dir, 'bisect-01', 'bisect.log')) as fin:
...         log_contents = fin.read()
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile
<BLANKLINE>
Updating ground-truth results - ground-truth.csv - done
Searching for bad source files:
  Created /.../bisect-01/bisect-make-01.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-02.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-03.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-04.mk - compiling and running - bad
    Found bad source file tests/file1.cpp
  Created /.../bisect-01/bisect-make-05.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-06.mk - compiling and running - bad
    Found bad source file tests/file3.cpp
  Created /.../bisect-01/bisect-make-07.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-08.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-09.mk - compiling and running - bad
    Found bad source file tests/file2.cpp
  Created /.../bisect-01/bisect-make-10.mk - compiling and running - good
  bad sources:
    tests/file1.cpp
    tests/file3.cpp
    tests/file2.cpp
Searching for bad symbols in: tests/file1.cpp
  Created /.../bisect-01/bisect-make-11.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-12.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-13.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-14.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-15.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-16.mk - compiling and running - bad
    Found bad symbol on line 9 -- file1_func2_PROBLEM()
  Created /.../bisect-01/bisect-make-17.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-18.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-19.mk - compiling and running - bad
    Found bad symbol on line 17 -- file1_func3_PROBLEM()
  Created /.../bisect-01/bisect-make-20.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-21.mk - compiling and running - bad
    Found bad symbol on line 25 -- file1_func4_PROBLEM()
  Created /.../bisect-01/bisect-make-22.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-23.mk - compiling and running - good
  bad symbols in tests/file1.cpp:
    line 9 -- file1_func2_PROBLEM()
    line 17 -- file1_func3_PROBLEM()
    line 25 -- file1_func4_PROBLEM()
Searching for bad symbols in: tests/file3.cpp
  Created /.../bisect-01/bisect-make-24.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-25.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-26.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-27.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-28.mk - compiling and running - bad
    Found bad symbol on line 9 -- file3_func2_PROBLEM()
  Created /.../bisect-01/bisect-make-29.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-30.mk - compiling and running - bad
    Found bad symbol on line 20 -- file3_func5_PROBLEM()
  Created /.../bisect-01/bisect-make-31.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-32.mk - compiling and running - good
  bad symbols in tests/file3.cpp:
    line 9 -- file3_func2_PROBLEM()
    line 20 -- file3_func5_PROBLEM()
Searching for bad symbols in: tests/file2.cpp
  Created /.../bisect-01/bisect-make-33.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-34.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-35.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-36.mk - compiling and running - bad
  Created /.../bisect-01/bisect-make-37.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-38.mk - compiling and running - bad
    Found bad symbol on line 8 -- file2_func1_PROBLEM()
  Created /.../bisect-01/bisect-make-39.mk - compiling and running - good
  Created /.../bisect-01/bisect-make-40.mk - compiling and running - good
  bad symbols in tests/file2.cpp:
    line 8 -- file2_func1_PROBLEM()
All bad symbols:
  /.../tests/file1.cpp:9 ... -- file1_func2_PROBLEM()
  /.../tests/file1.cpp:17 ... -- file1_func3_PROBLEM()
  /.../tests/file1.cpp:25 ... -- file1_func4_PROBLEM()
  /.../tests/file3.cpp:9 ... -- file3_func2_PROBLEM()
  /.../tests/file3.cpp:20 ... -- file3_func5_PROBLEM()
  /.../tests/file2.cpp:8 ... -- file2_func1_PROBLEM()

TODO: test the log_contents variable
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
