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
Tests the use of the compiler-specific flags in flit-config.toml for both the
compilation under test and for the link step (to use the baseline compilation)

>>> import glob
>>> import os
>>> import shutil
>>> import subprocess as subp
>>> from io import StringIO
>>> import flitutil as util

>>> class BisectTestError(RuntimeError): pass

>>> def bisect_compile(compiler, directory):
...     'Runs bisect with the given compiler and returns makefile variables'
...
...     # create static variable i for how many times this function is called
...     if not hasattr(bisect_compile, 'i'):
...         bisect_compile.i = 0  # initialize only once
...     bisect_compile.i += 1
...
...     # Note: we give a bad flag to cause bisect to error out early
...     #   we just need it to get far enough to create the first makefile
...     with StringIO() as ostream:
...         retval = th.flit.main(['bisect', '-C', directory,
...                                '--precision', 'double',
...                                compiler + ' -bad-flag', 'EmptyTest'],
...                                outstream=ostream)
...         if retval == 0:
...             raise BisectTestError('Expected bisect to fail\\n' +
...                                   ostream.getvalue())
...
...     # Since each call creates a separate bisect dir, we use our counter
...     makevars = util.extract_make_vars(
...         makefile=os.path.join(directory,
...                               'bisect-{0:02d}'.format(bisect_compile.i),
...                               'bisect-make-01.mk'),
...         directory=directory)
...     return makevars

>>> with th.tempdir() as temp_dir:
...     with StringIO() as ostream:
...         retval = th.flit.main(['init', '-C', temp_dir], outstream=ostream)
...         if retval != 0:
...             raise BisectTestError(
...                 'Could not initialize (retval={0}):\\n'.format(retval) +
...                 ostream.getvalue())
...     _ = shutil.copy(
...         os.path.join('data', 'flit-config-compilerspecificflags.toml'),
...         os.path.join(temp_dir, 'flit-config.toml'))
...     _ = shutil.copy(os.path.join('data', 'fake_gcc4.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_gcc9.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_clang34.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_intel19.py'), temp_dir)
...     with StringIO() as ostream:
...         retval = th.flit.main(['update', '-C', temp_dir],
...                               outstream=ostream)
...         if retval != 0:
...             raise BisectTestError('Could not update Makefile\\n' +
...                                   ostream.getvalue())
...
...     bisect_makevars_gcc4 = bisect_compile('./fake_gcc4.py', temp_dir)
...     bisect_makevars_gcc9 = bisect_compile('./fake_gcc9.py', temp_dir) # not in flitconfig
...     bisect_makevars_clang = bisect_compile('./fake_clang34.py', temp_dir)
...     bisect_makevars_intel = bisect_compile('./fake_intel19.py', temp_dir)

Note: fake_gcc9.py is not in the flit-config.toml, and therefore should not
  have fixed flags pulled from it.
>>> bisect_makevars_gcc4['GT_SWITCHES']
['-W-baseline-flag1']
>>> sorted(bisect_makevars_gcc4['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(bisect_makevars_clang['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(bisect_makevars_intel['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(bisect_makevars_gcc9['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']

>>> sorted(bisect_makevars_gcc4['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_clang['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_intel['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_gcc9['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']

>>> sorted(bisect_makevars_gcc4['TROUBLE_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2']
>>> sorted(bisect_makevars_clang['TROUBLE_CXXFLAGS'])
['-W-clang-flag1', '-W-clang-flag2']
>>> sorted(bisect_makevars_intel['TROUBLE_CXXFLAGS'])
['-W-intel-flag1', '-W-intel-flag2']
>>> sorted(bisect_makevars_gcc9['TROUBLE_CXXFLAGS'])
[]

>>> sorted(bisect_makevars_gcc4['TROUBLE_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_clang['TROUBLE_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(bisect_makevars_intel['TROUBLE_LDFLAGS'])
['-l-intel-link1', '-l-intel-link2', '-no-pie']
>>> sorted(bisect_makevars_gcc9['TROUBLE_LDFLAGS'])
[]

>>> sorted(bisect_makevars_gcc4['BISECT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_clang['BISECT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_intel['BISECT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(bisect_makevars_gcc9['BISECT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
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
