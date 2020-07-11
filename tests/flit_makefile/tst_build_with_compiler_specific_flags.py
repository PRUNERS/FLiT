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
Tests FLiT's use of the 'fixed_compile_flags' and 'fixed_link_flags' from the
compiler section of the flit-config.toml file

>>> from io import StringIO
>>> import glob
>>> import os
>>> import shutil
>>> import subprocess as subp

Delete MAKEFLAGS so that silent mode does not propogate
>>> if 'MAKEFLAGS' in os.environ:
...     del os.environ['MAKEFLAGS']

>>> with th.tempdir() as temp_dir:
...     with StringIO() as ostream:
...         _ = th.flit.main(['init', '-C', temp_dir], outstream=ostream)
...         init_out = ostream.getvalue().splitlines()
...     os.remove(os.path.join(temp_dir, 'flit-config.toml'))
...     with open(os.path.join(temp_dir, 'flit-config.toml'), 'w') as conf:
...         _ = conf.write('[dev_build]\\n')
...         _ = conf.write("compiler_name = 'fake-clang'\\n")
...         _ = conf.write('[ground_truth]\\n')
...         _ = conf.write("compiler_name = 'fake-clang'\\n")
...         _ = conf.write('[[compiler]]\\n')
...         _ = conf.write("binary = './fake_clang34.py'\\n")
...         _ = conf.write("name = 'fake-clang'\\n")
...         _ = conf.write("type = 'clang'\\n")
...         _ = conf.write("fixed_compile_flags = '-Wfixed-flag'\\n")
...         _ = conf.write("fixed_link_flags = '-Wfix-link -Whello'\\n")
...     _ = shutil.copy('fake_clang34.py', temp_dir)
...     _ = subp.check_output(['make', '--always-make', 'Makefile',
...                            '-C', temp_dir])
...     make_out = subp.check_output(['make', 'gt', '-C', temp_dir,
...                                   'VERBOSE=1'])
...     make_out = make_out.decode('utf8').splitlines()

Verify the output of flit init
>>> print('\\n'.join(init_out)) # doctest:+ELLIPSIS
Creating .../flit-config.toml
Creating .../custom.mk
Creating .../main.cpp
Creating .../tests/Empty.cpp
Creating .../Makefile

>>> sum([1 for x in make_out if '-Wfixed-flag' in x])
5

>>> sum([1 for x in make_out if '-Wfix-link -Whello' in x])
2

Make sure gcc toolchain is not used since gcc is not there
>>> any(['--gcc-toolchain' in x for x in make_out])
False
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

