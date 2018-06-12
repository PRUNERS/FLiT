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
Tests FLiT's capabilities to create a functioning empty project

The tests are below using doctest

Let's now make a temporary directory and test that flit init populates the
correct files
>>> import glob
>>> import os
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     files = os.listdir(temp_dir)
Creating /.../flit-config.toml
Creating /.../custom.mk
Creating /.../main.cpp
Creating /.../tests/Empty.cpp
Creating /.../Makefile
>>> s = set(['flit-config.toml', 'custom.mk', 'main.cpp', 'tests', 'Makefile'])
>>> s.issubset(files)
True

Now we want to specifically test the Makefile and its capabilities for the
empty project.

The following test checks that the dev job is mostly correct.  It may be a
little too forgiving, but making it too restrictive makes changing the code
very difficult.

>>> import subprocess as subp
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     actual = subp.check_output(['make', '-C', temp_dir, '-n', 'dev'])
Creating ...
>>> actual = actual.decode('utf-8').strip()
>>> '-DFLIT_HOST=' in actual
True
>>> '-DFLIT_COMPILER=\\'"g++"\\'' in actual
True
>>> '-DFLIT_OPTL=\\'"-O2"\\'' in actual
True
>>> '-DFLIT_SWITCHES=\\'"-funsafe-math-optimizations"\\'' in actual
True
>>> '-DFLIT_FILENAME=\\'"devrun"\\'' in actual
True
>>> '-o devrun obj/main_dev.o obj/Empty_dev.o' in actual
True
>>> '-lm -lstdc++ -L{libdir} -lflit -Wl,-rpath={libdir}' \\
... .format(libdir=th.config.lib_dir) in actual
True

Let's actually now compile and run the empty test under different circumstances

TODO: this takes too long to run.  Can we do something faster?

>>> import subprocess as subp
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     _ = subp.check_output(['make', '-C', temp_dir, 'dev', '-j8'],
...                           stderr=subp.STDOUT)
...     devrun_out = subp.check_output([os.path.join(temp_dir, 'devrun')],
...                                    stderr=subp.STDOUT)
...     print(devrun_out.decode('utf-8'), end='')
Creating ...
name,host,compiler,optl,switches,precision,score_hex,score,resultfile,comparison_hex,comparison,file,nanosec

TODO: let's test that the full set of tests would get created.
TODO: test that the custom.mk stuff gets used
TODO: test CLANG_ONLY
TODO: test targets dev, gt, run, runbuild, clean, veryclean, distclean
TODO: test solution on a mac (with UNAME_S being Darwin)
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
