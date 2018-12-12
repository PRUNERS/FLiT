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
Tests only providing gcc and no other compiler.  Tests that
- the binary is taken from the configuration file
- the other two compilers are not present (i.e. clang and intel)

>>> from io import StringIO
>>> import os
>>> import shutil

>>> testconf = 'data/onlyprovidedcompilers.toml'
>>> class UpdateTestError(RuntimeError): pass

>>> def deref_makelist(name):
...     return sorted([' '.join(makevars[x]) for x in makevars[name]])
>>> def get_default_compiler(typename):
...     defaults = th.util.get_default_toml()
...     default_compiler = [x for x in defaults['compiler']
...                         if x['type'] == typename]
...     assert len(default_compiler) == 1
...     return default_compiler[0]

>>> with th.tempdir() as temp_dir:
...     with StringIO() as ostream:
...         retval = th.flit.main(['init', '-C', temp_dir], outstream=ostream)
...         init_out = ostream.getvalue().splitlines()
...     if retval != 0:
...         raise UpdateTestError('Failed to initialize flit directory')
...     _ = shutil.copy(testconf, os.path.join(temp_dir, 'flit-config.toml'))
...     with StringIO() as ostream:
...         retval = th.flit.main(['update', '-C', temp_dir],
...                               outstream=ostream)
...         update_out = ostream.getvalue().splitlines()
...     if retval != 0:
...         raise UpdateTestError('Failed to update Makefile')
...     makevars = th.util.extract_make_vars(directory=temp_dir)

Get default values for each compiler from the default configuration

>>> print('\\n'.join(init_out)) # doctest:+ELLIPSIS
Creating .../flit-config.toml
Creating .../custom.mk
Creating .../main.cpp
Creating .../tests/Empty.cpp
Creating .../Makefile

>>> print('\\n'.join(update_out)) # doctest:+ELLIPSIS
Updating .../Makefile

Check that the compiler binaries are the default values
>>> makevars['GCC']
['/usr/bin/my-g++']

>>> makevars['CLANG']
['None']

>>> makevars['INTEL']
['None']

>>> makevars['COMPILERS']
['GCC']

>>> 'OPCODES_GCC' in makevars
True
>>> 'SWITCHES_GCC' in makevars
True
>>> 'OPCODES_CLANG' in makevars
False
>>> 'SWITCHES_CLANG' in makevars
False
>>> 'OPCODES_INTEL' in makevars
False
>>> 'SWITCHES_INTEL' in makevars
False
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
