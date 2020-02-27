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
Tests FLiT Bisect, specifically the command-line options related to linking.

>>> import os
>>> import shutil
>>> from common import bisect_compile, flit_init, flit_update

>>> def dual_bisect_compile(*args, add_ldflags=None, **kwargs):
...     without = bisect_compile(*args, **kwargs)
...     added = bisect_compile(*args, add_ldflags=add_ldflags, **kwargs)
...     return without, added

>>> with th.tempdir() as temp_dir:
...     _ = flit_init(temp_dir)
...     _ = shutil.copy(
...         os.path.join('data', 'flit-config-compilerspecificflags.toml'),
...         os.path.join(temp_dir, 'flit-config.toml'))
...     _ = shutil.copy(os.path.join('data', 'fake_gcc4.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_gcc9.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_clang34.py'), temp_dir)
...     _ = shutil.copy(os.path.join('data', 'fake_intel19.py'), temp_dir)
...     flit_update(temp_dir)
...
...     # Note: ./fake_gcc9.py is not in flit-config.toml
...     vars_ldflags, vars_ldflags_added = dual_bisect_compile(
...         './fake_gcc4.py', temp_dir, ldflags='-other -ldflags',
...         add_ldflags='-add1 -add2')
...     vars_clanglink, vars_clanglink_added = dual_bisect_compile(
...         './fake_intel19.py', temp_dir, linker='./fake_clang34.py',
...         add_ldflags='-add3')
...     vars_both, vars_both_added = dual_bisect_compile(
...         './fake_clang34.py', temp_dir, linker='./fake_intel19.py',
...         ldflags='-more -still', add_ldflags='-add4 -add5')
...     vars_unknown_linker, vars_unknown_linker_added = dual_bisect_compile(
...         './fake_clang34.py', temp_dir, linker='./fake_gcc9.py',
...         add_ldflags='-add6 -add7 -add8')
...     vars_unknown_linker_ld, vars_unknown_linker_ld_added = dual_bisect_compile(
...         './fake_gcc4.py', temp_dir, linker='./fake_gcc9.py',
...         ldflags='-ld-for-unkonwn', add_ldflags='')

Test that BISECT_LDFLAGS is populated correctly
-----------------------------------------------

Uses the ld flags that were given
>>> sorted(vars_ldflags['BISECT_LDFLAGS'])
['-ldflags', '-other']
>>> sorted(vars_ldflags_added['BISECT_LDFLAGS'])
['-add1', '-add2', '-ldflags', '-other']

Use flags from ./fake_clang34.py
>>> sorted(vars_clanglink['BISECT_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(vars_clanglink_added['BISECT_LDFLAGS'])
['-add3', '-l-clang-link1', '-l-clang-link2', '-nopie']

Use the ld flags that were given
>>> sorted(vars_both['BISECT_LDFLAGS'])
['-more', '-still']
>>> sorted(vars_both_added['BISECT_LDFLAGS'])
['-add4', '-add5', '-more', '-still']

Empty because linker is not found in flit-config.toml
>>> sorted(vars_unknown_linker['BISECT_LDFLAGS'])
[]
>>> sorted(vars_unknown_linker_added['BISECT_LDFLAGS'])
['-add6', '-add7', '-add8']

Use the ld flags that were given
>>> sorted(vars_unknown_linker_ld['BISECT_LDFLAGS'])
['-ld-for-unknown']
>>> sorted(vars_unknown_linker_ld_added['BISECT_LDFLAGS'])
['-ld-for-unknown']

Test that the BISECT_LINK is set appropriately
--------------------------------------------

>>> sorted(vars_ldflags['BISECT_LINK'])
['./fake_gcc4.py']
>>> sorted(vars_ldflags_added['BISECT_LINK'])
['./fake_gcc4.py']
>>> sorted(vars_clanglink['BISECT_LINK'])
['./fake_clang34.py']
>>> sorted(vars_clanglink_added['BISECT_LINK'])
['./fake_clang34.py']
>>> sorted(vars_both['BISECT_LINK'])
['./fake_intel19.py']
>>> sorted(vars_both_added['BISECT_LINK'])
['./fake_intel19.py']
>>> sorted(vars_unknown_linker['BISECT_LINK'])
['./fake_gcc9.py']
>>> sorted(vars_unknown_linker_added['BISECT_LINK'])
['./fake_gcc9.py']
>>> sorted(vars_unknown_linker_ld['BISECT_LINK'])
['./fake_gcc9.py']
>>> sorted(vars_unknown_linker_ld_added['BISECT_LINK'])
['./fake_gcc9.py']

Test GT_CXXFLAGS is unaffected by extra flags
---------------------------------------------

>>> sorted(vars_ldflags['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_ldflags_added['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_clanglink['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_clanglink_added['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_both['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_both_added['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_unknown_linker['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_unknown_linker_added['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_unknown_linker_ld['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']
>>> sorted(vars_unknown_linker_ld_added['GT_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2', '-g']

Test GT_LDFLAGS is unaffected by extra flags
--------------------------------------------

>>> sorted(vars_ldflags['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_ldflags_added['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_clanglink['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_clanglink_added['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_both['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_both_added['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_unknown_linker['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_unknown_linker_added['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_unknown_linker_ld['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_unknown_linker_ld_added['GT_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']

Test TROUBLE_CXXFLAGS is unaffected by extra flags
--------------------------------------------------

>>> sorted(vars_ldflags['TROUBLE_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2']
>>> sorted(vars_ldflags_added['TROUBLE_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2']
>>> sorted(vars_clanglink['TROUBLE_CXXFLAGS'])
['-W-intel-flag1', '-W-intel-flag2']
>>> sorted(vars_clanglink_added['TROUBLE_CXXFLAGS'])
['-W-intel-flag1', '-W-intel-flag2']
>>> sorted(vars_both['TROUBLE_CXXFLAGS'])
['-W-clang-flag1', '-W-clang-flag2']
>>> sorted(vars_both_added['TROUBLE_CXXFLAGS'])
['-W-clang-flag1', '-W-clang-flag2']
>>> sorted(vars_unknown_linker['TROUBLE_CXXFLAGS'])
['-W-clang-flag1', '-W-clang-flag2']
>>> sorted(vars_unknown_linker_added['TROUBLE_CXXFLAGS'])
['-W-clang-flag1', '-W-clang-flag2']
>>> sorted(vars_unknown_linker_ld['TROUBLE_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2']
>>> sorted(vars_unknown_linker_ld_added['TROUBLE_CXXFLAGS'])
['-W-gcc-flag1', '-W-gcc-flag2']

Test TROUBLE_LDFLAGS is unaffected by extra flags
-------------------------------------------------

>>> sorted(vars_ldflags['TROUBLE_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_ldflags_added['TROUBLE_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_clanglink['TROUBLE_LDFLAGS'])
['-l-intel-link1', '-l-intel-link2', '-no-pie']
>>> sorted(vars_clanglink_added['TROUBLE_LDFLAGS'])
['-l-intel-link1', '-l-intel-link2', '-no-pie']
>>> sorted(vars_both['TROUBLE_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(vars_both_added['TROUBLE_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(vars_unknown_linker['TROUBLE_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(vars_unknown_linker_added['TROUBLE_LDFLAGS'])
['-l-clang-link1', '-l-clang-link2', '-nopie']
>>> sorted(vars_unknown_linker_ld['TROUBLE_LDFLAGS'])
['-l-gcc-link1', '-l-gcc-link2']
>>> sorted(vars_unknown_linker_ld_added['TROUBLE_LDFLAGS'])
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
