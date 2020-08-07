# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
#   https://github.com/PRUNERS/FLiT/blob/main/LICENSE
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

>>> import os
>>> import shutil
>>> import subprocess as subp
>>> from io import StringIO
>>> from common import BisectTestError, flit_init

>>> with th.tempdir() as temp_dir:
...     init_out = flit_init(temp_dir)
...     shutil.rmtree(os.path.join(temp_dir, 'tests'))
...     _ = shutil.copytree(os.path.join('data', 'tests'),
...                         os.path.join(temp_dir, 'tests'))
...     _ = shutil.copy(os.path.join('data',
...                                  'auto-bisect-run-flit-config.toml'),
...                     os.path.join(temp_dir, 'flit-config.toml'))
...     _ = shutil.copy(os.path.join('..', '..', 'flit_makefile',
...                                  'fake_clang34.py'),
...                     temp_dir)
...     with open(os.path.join(temp_dir, 'custom.mk'), 'a') as mkout:
...         _ = mkout.write('SOURCE += tests/file4.cxx\\n')
...     with StringIO() as ostream:
...         retval = th.flit.main(['import', '--dbfile',
...                                os.path.join(temp_dir, 'autorun.sqlite'),
...                                os.path.join('data', 'auto-bisect-run.csv')],
...                               outstream=ostream)
...         if retval != 0:
...             raise BisectTestError(
...                 'Could not import (retval={0}):\\n'.format(retval) +
...                 ostream.getvalue())
...     with StringIO() as ostream:
...         retval = th.flit.main(['bisect', '--directory', temp_dir,
...                                '--auto-sqlite-run',
...                                os.path.join(temp_dir, 'autorun.sqlite'),
...                                '--compile-only'],
...                               outstream=ostream)
...         if retval != 0:
...             raise BisectTestError(
...                 'Could not bisect (retval={0}):\\n'.format(retval) +
...                 ostream.getvalue())
...         bisect_out = ostream.getvalue().splitlines()
...     makeout1 = subp.check_output([
...         'make', '-C', temp_dir, '--dry-run', 'trouble',
...         '--no-print-directory', '--always-make',
...         '-f', os.path.join('bisect-precompile', 'bisect-make-01.mk')])
...     makeout1 = makeout1.strip().decode('utf-8').splitlines()
...     makeout2 = subp.check_output([
...         'make', '-C', temp_dir, '--dry-run', 'gt',
...         '--no-print-directory', '--always-make',
...         '-f', os.path.join('bisect-precompile', 'bisect-make-01.mk')])
...     makeout2 = makeout2.strip().decode('utf-8').splitlines()
...     makeout3 = subp.check_output([
...         'make', '-C', temp_dir, '--dry-run', 'bisect-smallclean',
...         '--no-print-directory', '--always-make',
...         '-f', os.path.join('bisect-precompile', 'bisect-make-01.mk')])
...     makeout3 = makeout3.strip().decode('utf-8').splitlines()
...     troublecxx = th.util.extract_make_var(
...         'TROUBLE_CXX',
...         os.path.join('bisect-precompile', 'bisect-make-01.mk'),
...         directory=temp_dir)
...     troublecxx_type = th.util.extract_make_var(
...         'TROUBLE_CXX_TYPE',
...         os.path.join('bisect-precompile', 'bisect-make-01.mk'),
...         directory=temp_dir)

Verify the output of flit init
>>> print('\\n'.join(init_out)) # doctest:+ELLIPSIS
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile

Let's make sure that the fake clang is not called with --gcc-toolchain
>>> fakeclang_lines = [line for line in makeout1
...                    if line.startswith('./fake_clang34.py')]
>>> len(fakeclang_lines)
8
>>> any('--gcc-toolchain' in line for line in fakeclang_lines)
False

Let's make sure that gcc is not called with --gcc-toolchain
>>> gcc_lines = [line for line in makeout2
...              if line.startswith('g++ ')]
>>> len(gcc_lines)
9
>>> any('--gcc-toolchain' in line for line in gcc_lines)
False

See that the generated Makefile for fake_clang34.py shows the correct type
>>> troublecxx
['./fake_clang34.py']
>>> troublecxx_type
['clang']

>>> 'rm -rf bisect-precompile/obj/split' in makeout3
True
>>> 'rm -rf bisect-precompile/obj/symbols' not in makeout3
True
>>> 'rm -rf bisect-precompile/obj/fpic' not in makeout3
True
>>> 'rm -rf bisect-precompile/obj' not in makeout3
True
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
