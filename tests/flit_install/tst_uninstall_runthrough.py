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
Tests FLiT's capabilities to uninstall itself

The tests are below using doctest

Let's now make a temporary directory, install there, and then uninstall.
>>> import glob
>>> import os
>>> import subprocess as subp
>>> with th.tempdir() as temp_dir:
...     _ = subp.check_call(['make', '-C', os.path.join(th.config.lib_dir, '..'),
...                          'install', 'PREFIX=' + temp_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     dirs1 = os.listdir(temp_dir)
...     dirs2 = [os.path.join(x, y) for x in dirs1
...                                 for y in os.listdir(os.path.join(temp_dir, x))]
...     _ = subp.check_call(['make', '-C', os.path.join(th.config.lib_dir, '..'),
...                          'uninstall', 'PREFIX=' + temp_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     tempdir_exists = os.path.exists(temp_dir)
>>> sorted(dirs1)
['bin', 'include', 'lib', 'share']
>>> sorted(dirs2)
['bin/flit', 'include/flit', 'lib/libflit.so', 'share/flit', 'share/licenses']
>>> tempdir_exists
False

Now we test that directories are not deleted when other files exist in the
PREFIX path
>>> import glob
>>> with th.tempdir() as temp_dir:
...     _ = subp.check_call(['make', '-C', os.path.join(th.config.lib_dir, '..'),
...                          'install', 'PREFIX=' + temp_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     with open(os.path.join(temp_dir, 'lib', 'otherlib.so'), 'w'):
...         pass
...     _ = subp.check_call(['make', '-C', os.path.join(th.config.lib_dir, '..'),
...                          'uninstall', 'PREFIX=' + temp_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     prevdir = os.path.realpath(os.curdir)
...     os.chdir(temp_dir)
...     all_files = [os.path.join(base, name)
...                  for base, folders, filenames in os.walk('.')
...                  for name in folders + filenames]
...     os.chdir(prevdir)
>>> sorted(all_files)
['./lib', './lib/otherlib.so']
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
