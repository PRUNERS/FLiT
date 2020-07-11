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
Tests error cases in the configuration file, such as specifying more than one of a certain type of compiler.

>>> from io import StringIO
>>> import os
>>> import shutil

>>> from tst_common_funcs import runconfig

>>> configstr = \\
...     '[dev_build]\\n' \\
...     'compiler_name = \\'name-does-not-exist\\'\\n'
>>> runconfig(configstr)
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: Compiler name name-does-not-exist not found

>>> configstr = \\
...     '[ground_truth]\\n' \\
...     'compiler_name = \\'another-name-that-does-not-exist\\'\\n'
>>> runconfig(configstr)
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: Compiler name another-name-that-does-not-exist not found

>>> runconfig('[compiler]\\n')
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml improperly configured, needs [[compiler]] section

>>> runconfig('[[compiler]]\\n')
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: compiler "{}" is missing the "name" field

>>> runconfig('[[compiler]]\\n'
...           'name = \\'hello\\'\\n')
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: compiler "{'name': 'hello'}" is missing the "type" field

>>> runconfig('[[compiler]]\\n'
...           'name = \\'hello\\'\\n'
...           'type = \\'gcc\\'\\n') # doctest:+ELLIPSIS
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: compiler "{...}" is missing the "binary" field

>>> runconfig('[[compiler]]\\n'
...           'binary = \\'my-special-compiler\\'\\n'
...           'name = \\'hello\\'\\n'
...           'type = \\'my-unsupported-type\\'\\n')
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: unsupported compiler type "my-unsupported-type"

>>> runconfig('[[compiler]]\\n'
...           'binary = \\'gcc\\'\\n'
...           'name = \\'gcc\\'\\n'
...           'type = \\'gcc\\'\\n'
...           '\\n'
...           '[[compiler]]\\n'
...           'binary = \\'gcc-2\\'\\n'
...           'name = \\'gcc-2\\'\\n'
...           'type = \\'gcc\\'\\n'
...           )
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: cannot have multiple compilers of the same type (gcc)

>>> runconfig('[[compiler]]\\n'
...           'binary = \\'gcc\\'\\n'
...           'name = \\'gcc\\'\\n'
...           'type = \\'gcc\\'\\n'
...           '\\n'
...           '[[compiler]]\\n'
...           'binary = \\'gcc-2\\'\\n'
...           'name = \\'gcc\\'\\n'
...           'type = \\'clang\\'\\n'
...           )
Traceback (most recent call last):
...
tst_common_funcs.UpdateTestError: Failed to update Makefile: Error: flit-config.toml: cannot have multiple compilers of the same name (gcc)
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
