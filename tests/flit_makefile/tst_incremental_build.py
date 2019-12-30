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
Tests FLiT's generated Makefile and its ability to correctly do incremental
builds.

The tests are below using doctest

>>> import glob
>>> import os
>>> import subprocess as subp
>>> import re
>>> import time

Delete MAKEFLAGS so that silent mode does not propogate
>>> if 'MAKEFLAGS' in os.environ:
...     del os.environ['MAKEFLAGS']

Test that the dev target will be rebuilt if one of the files is updated.
We use 'make --touch' to simply touch files it would create and then we will
test that the correct files are updated.
>>> def compile_target(directory, target, touch=True):
...     'Compiles the dev target using "make --touch" and returns the output'
...     command = ['make', '-C', directory, target, 'VERBOSE=1']
...     if touch:
...         command.append('--touch')
...     output = subp.check_output(command)
...     return output.decode('utf-8').strip()
>>> compile_dev = lambda x: compile_target(x, 'dev')
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     # fake the built elements -- don't actually build for time's sake
...     _ = compile_target(temp_dir, 'dirs', touch=False)
...     before_build = compile_dev(temp_dir)
...     # this next build should say nothing to do
...     after_build = compile_dev(temp_dir)
...     # update one file and recompile
...     with open(os.path.join(temp_dir, 'main.cpp'), 'a') as mainfile:
...         mainfile.write('#include "new_header.h"\\n')
...     maindepname = os.path.join(temp_dir, 'obj', 'dev', 'main.cpp.d')
...     with open(maindepname, 'w') as maindep:
...         maindep.write('obj/dev/main.cpp.o: main.cpp new_header.h\\n')
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_modify = compile_dev(temp_dir)
...     # touch the header file and make sure it recompiles again
...     time.sleep(0.01) # give some time before touching again
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_touch = compile_dev(temp_dir)
Creating ...

>>> def touched_files(outstring):
...     'Returns list of touched files in sorted order'
...     return sorted({x[6:] for x in outstring.splitlines()
...                    if x.startswith('touch ')})

Make sure all of the correct files were created with our build commands

>>> touched_files(before_build)
['devrun', 'obj/dev/Empty.cpp.o', 'obj/dev/main.cpp.o']

>>> touched_files(after_build)
[]

>>> touched_files(after_modify)
['devrun', 'obj/dev/main.cpp.o']

>>> touched_files(after_touch)
['devrun', 'obj/dev/main.cpp.o']

Now, let's test the same thing with the "gt" target

>>> compile_gt = lambda x: compile_target(x, 'gt')
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     # fake the built elements -- don't actually build for time's sake
...     compile_target(temp_dir, 'dirs', touch=False)
...     before_build = compile_gt(temp_dir)
...     # this next build should say nothing to do
...     after_build = compile_gt(temp_dir)
...     # update one file and recompile
...     with open(os.path.join(temp_dir, 'main.cpp'), 'a') as mainfile:
...         mainfile.write('#include "new_header.h"\\n')
...     maindepname = os.path.join(temp_dir, 'obj', 'gt', 'main.cpp.d')
...     with open(maindepname, 'w') as maindep:
...         maindep.write('obj/gt/main.cpp.o: main.cpp new_header.h\\n')
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_modify = compile_gt(temp_dir)
...     # touch the header file and make sure it recompiles again
...     time.sleep(0.01) # give some time before touching again
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_touch = compile_gt(temp_dir)
Creating ...

Make sure all of the correct files were created with our build commands

>>> touched_files(before_build)
['gtrun', 'obj/gt/Empty.cpp.o', 'obj/gt/main.cpp.o']

>>> touched_files(after_build)
[]

>>> touched_files(after_modify)
['gtrun', 'obj/gt/main.cpp.o']

>>> touched_files(after_touch)
['gtrun', 'obj/gt/main.cpp.o']
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
