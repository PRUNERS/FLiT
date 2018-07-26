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
Tests FLiT's capabilities to compile and run MPI tests

The tests are below using doctest

Let's now make a temporary directory and test that we can successfully compile
and run the FLiT test with MPI

>>> import glob
>>> import os
>>> import shutil
>>> import subprocess as subp

>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     _ = shutil.copy(os.path.join('data', 'MpiHello.cpp'),
...                     os.path.join(temp_dir, 'tests'))
...     _ = shutil.copy(os.path.join('data', 'flit-config.toml'), temp_dir)
...     th.flit.main(['update', '-C', temp_dir])
...     compile_str = subp.check_output(['make', '-C', temp_dir, 'gt'],
...                                     stderr=subp.STDOUT)
...     run_str = subp.check_output(['make', '-C', temp_dir, 'ground-truth.csv'],
...                                 stderr=subp.STDOUT)
...     file_f = os.path.join(temp_dir, 'ground-truth.csv_MpiHello_f.dat')
...     file_d = os.path.join(temp_dir, 'ground-truth.csv_MpiHello_d.dat')
...     file_e = os.path.join(temp_dir, 'ground-truth.csv_MpiHello_e.dat')
...     with open(file_f, 'r') as fin: contents_f = fin.read()
...     with open(file_d, 'r') as fin: contents_d = fin.read()
...     with open(file_e, 'r') as fin: contents_e = fin.read()
...     run_str_2 = subp.check_output([
...         'mpirun', '-n', '2',
...         os.path.join(temp_dir, 'gtrun'),
...         '--precision', 'double',
...         '--output', os.path.join(temp_dir, 'ground-truth.csv'),
...         '--timing-repeats', '1',
...         '--timing-loops', '-1',
...         ], stderr=subp.STDOUT)
...     run_str_3 = subp.check_output([
...         'mpirun', '-n', '1',
...         os.path.join(temp_dir, 'gtrun'),
...         '--precision', 'double',
...         '--output', os.path.join(temp_dir, 'ground-truth.csv'),
...         '--timing-repeats', '1',
...         '--timing-loops', '-1',
...         ], stderr=subp.STDOUT)
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile
>>> compile_str = compile_str.decode('utf-8').strip().splitlines()
>>> run_str = run_str.decode('utf-8').strip().splitlines()
>>> run_str_2 = run_str_2.decode('utf-8').strip().splitlines()
>>> run_str_3 = run_str_3.decode('utf-8').strip().splitlines()

Make sure the info statement about MPI being enabled is done at each make call

>>> 'MPI is enabled' in compile_str
True
>>> 'MPI is enabled' in run_str
True

Make sure the correct arguments are passed to mpirun

>>> 'mpirun -n 2 ./gtrun --output ground-truth.csv --no-timing' in run_str
True

Make sure the console messages are there, but they can be out of order

>>> run_str.count('MpiHello: hello from rank 0 of 2')
3
>>> run_str.count('MpiHello: hello from rank 1 of 2')
3

The string output files should only be written from rank 0, not rank 1

>>> contents_f
'MpiHello: hello from rank 0 of 2\\n'
>>> contents_d
'MpiHello: hello from rank 0 of 2\\n'
>>> contents_e
'MpiHello: hello from rank 0 of 2\\n'

Run #2 was with 2 mpi processes only running the test with double precision.
First make sure that the warning about looping only once is issued.

>>> run_str_2.count('Warning: cannot run auto-looping with MPI; Looping set to 1')
1

Now make sure the test was only run once

>>> run_str_2.count('MpiHello: hello from rank 0 of 2')
1
>>> run_str_2.count('MpiHello: hello from rank 1 of 2')
1

Run #3 was with 1 mpi process only running the test with double precision.
First make sure that the warning about looping was NOT issued.

>>> run_str_3.count('Warning: cannot run auto-looping with MPI; Looping set to 1')
0

Make sure the test was run multiple times

>>> run_str_3.count('MpiHello: hello from rank 0 of 1') > 1
True
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
