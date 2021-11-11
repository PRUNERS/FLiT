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
Tests FLiT's capabilities to run simple commands on an installation

The tests are below using doctest

Let's now make a temporary directory and install there.  Here we are simply
testing that the following commands complete without error.
>>> import glob
>>> import os
>>> import subprocess as subp
>>> import sys
>>> import shutil
>>> from importlib.machinery import SourceFileLoader
>>> with th.tempdir() as temp_dir:
...     destdir = os.path.join(temp_dir, 'install')
...     prefix = os.path.join(temp_dir, 'usr')
...     effective_prefix = os.path.join(destdir, prefix[1:])
...     _ = subp.check_call(['make',
...                          '-C', os.path.join(th.config.doc_dir, '..'),
...                          'install',
...                          'DESTDIR=' + destdir,
...                          'PREFIX=' + prefix],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     install_list = glob.glob(effective_prefix + '/**', recursive=True)
...     flit = os.path.join(effective_prefix, 'bin', 'flit')
...     sandbox_dir = os.path.join(effective_prefix, 'sandbox')
...     _ = subp.check_call([flit, 'init', '-C', sandbox_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     _ = subp.check_call(['make', '-C', sandbox_dir, 'dirs'],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     _ = subp.check_call(['make', '-C', sandbox_dir, '--touch', 'run'],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     os.chdir(sandbox_dir)
...     _ = subp.check_call([flit, 'import'] + glob.glob('results/*.csv'),
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     _ = shutil.copytree(effective_prefix, prefix, symlinks=True)
...     loader = SourceFileLoader(
...         'flitconfig',
...         os.path.join(prefix, 'share', 'flit', 'scripts', 'flitconfig.py'))
...     flitconfig = loader.load_module()

Make sure flitconfig has the correct paths

>>> flitconfig.script_dir == os.path.join(prefix, 'share', 'flit', 'scripts')
True

>>> flitconfig.doc_dir == os.path.join(prefix, 'share', 'flit', 'doc')
True

>>> flitconfig.src_dir == os.path.join(prefix, 'share', 'flit', 'src')
True

>>> flitconfig.include_dir == os.path.join(prefix, 'include')
True

>>> flitconfig.config_dir == os.path.join(prefix, 'share', 'flit', 'config')
True

>>> flitconfig.bash_completion_dir == os.path.join(
...     prefix, 'share', 'bash-completion', 'completions')
True

>>> flitconfig.data_dir == os.path.join(prefix, 'share', 'flit', 'data')
True

>>> flitconfig.litmus_test_dir == os.path.join(prefix, 'share', 'flit',
...                                            'litmus-tests')
True

Check that the expected list of installed files are all there
>>> all(x.startswith(effective_prefix) for x in install_list)
True

>>> from pprint import pprint
>>> pprint(sorted(x[len(effective_prefix)+1:] for x in install_list))
['',
 'bin',
 'bin/flit',
 'include',
 'include/flit',
 'include/flit.h',
 'include/flit/FlitCsv.h',
 'include/flit/InfoStream.h',
 'include/flit/TestBase.h',
 'include/flit/Variant.h',
 'include/flit/flit.h',
 'include/flit/flitHelpers.h',
 'include/flit/fsutil.h',
 'include/flit/subprocess.h',
 'include/flit/timeFunction.h',
 'include/flit/tinydir.h',
 'share',
 'share/bash-completion',
 'share/bash-completion/completions',
 'share/bash-completion/completions/flit',
 'share/flit',
 'share/flit/benchmarks',
 'share/flit/benchmarks/README.md',
 'share/flit/benchmarks/polybench',
 'share/flit/benchmarks/polybench/README.md',
 'share/flit/benchmarks/polybench/custom.mk',
 'share/flit/benchmarks/polybench/flit-config.toml',
 'share/flit/benchmarks/polybench/main.cpp',
 'share/flit/benchmarks/polybench/tests',
 'share/flit/benchmarks/polybench/tests/adi.cpp',
 'share/flit/benchmarks/polybench/tests/atax.cpp',
 'share/flit/benchmarks/polybench/tests/bicg.cpp',
 'share/flit/benchmarks/polybench/tests/cholesky.cpp',
 'share/flit/benchmarks/polybench/tests/correlation.cpp',
 'share/flit/benchmarks/polybench/tests/covariance.cpp',
 'share/flit/benchmarks/polybench/tests/deriche.cpp',
 'share/flit/benchmarks/polybench/tests/doitgen.cpp',
 'share/flit/benchmarks/polybench/tests/durbin.cpp',
 'share/flit/benchmarks/polybench/tests/fdtd_2d.cpp',
 'share/flit/benchmarks/polybench/tests/floyd_warshall.cpp',
 'share/flit/benchmarks/polybench/tests/gemm.cpp',
 'share/flit/benchmarks/polybench/tests/gemver.cpp',
 'share/flit/benchmarks/polybench/tests/gesummv.cpp',
 'share/flit/benchmarks/polybench/tests/gramschmidt.cpp',
 'share/flit/benchmarks/polybench/tests/heat_3d.cpp',
 'share/flit/benchmarks/polybench/tests/jacobi_1d.cpp',
 'share/flit/benchmarks/polybench/tests/jacobi_2d.cpp',
 'share/flit/benchmarks/polybench/tests/lu.cpp',
 'share/flit/benchmarks/polybench/tests/ludcmp.cpp',
 'share/flit/benchmarks/polybench/tests/mvt.cpp',
 'share/flit/benchmarks/polybench/tests/nussinov.cpp',
 'share/flit/benchmarks/polybench/tests/polybench_utils.h',
 'share/flit/benchmarks/polybench/tests/seidel_2d.cpp',
 'share/flit/benchmarks/polybench/tests/symm.cpp',
 'share/flit/benchmarks/polybench/tests/syr2k.cpp',
 'share/flit/benchmarks/polybench/tests/syrk.cpp',
 'share/flit/benchmarks/polybench/tests/test2mm.cpp',
 'share/flit/benchmarks/polybench/tests/test3mm.cpp',
 'share/flit/benchmarks/polybench/tests/trisolv.cpp',
 'share/flit/benchmarks/polybench/tests/trmm.cpp',
 'share/flit/benchmarks/random',
 'share/flit/benchmarks/random/README.md',
 'share/flit/benchmarks/random/custom.mk',
 'share/flit/benchmarks/random/flit-config.toml',
 'share/flit/benchmarks/random/main.cpp',
 'share/flit/benchmarks/random/tests',
 'share/flit/benchmarks/random/tests/Rand.cpp',
 'share/flit/benchmarks/random/tests/Random.cpp',
 'share/flit/config',
 'share/flit/config/flit-default.toml.in',
 'share/flit/config/version.txt',
 'share/flit/data',
 'share/flit/data/Makefile.in',
 'share/flit/data/Makefile_bisect_binary.in',
 'share/flit/data/custom.mk',
 'share/flit/data/db',
 'share/flit/data/db/tables-sqlite.sql',
 'share/flit/data/main.cpp',
 'share/flit/data/tests',
 'share/flit/data/tests/Empty.cpp',
 'share/flit/doc',
 'share/flit/doc/README.md',
 'share/flit/doc/analyze-results.md',
 'share/flit/doc/autogenerated-tests.md',
 'share/flit/doc/available-compiler-flags.md',
 'share/flit/doc/benchmarks.md',
 'share/flit/doc/compiling-your-tests.md',
 'share/flit/doc/cuda-support.md',
 'share/flit/doc/database-structure.md',
 'share/flit/doc/experimental-features.md',
 'share/flit/doc/flit-command-line.md',
 'share/flit/doc/flit-configuration-file.md',
 'share/flit/doc/flit-helpers.md',
 'share/flit/doc/installation.md',
 'share/flit/doc/litmus-tests.md',
 'share/flit/doc/mpi-support.md',
 'share/flit/doc/release-notes.md',
 'share/flit/doc/run-wrapper-and-hpc-support.md',
 'share/flit/doc/standard-c++-library-implementations.md',
 'share/flit/doc/test-executable.md',
 'share/flit/doc/test-input-generator.md',
 'share/flit/doc/writing-test-cases.md',
 'share/flit/litmus-tests',
 'share/flit/litmus-tests/DistributivityOfMultiplication.cpp',
 'share/flit/litmus-tests/DoHariGSBasic.cpp',
 'share/flit/litmus-tests/DoHariGSImproved.cpp',
 'share/flit/litmus-tests/DoMatrixMultSanity.cpp',
 'share/flit/litmus-tests/DoOrthoPerturbTest.cpp',
 'share/flit/litmus-tests/DoSimpleRotate90.cpp',
 'share/flit/litmus-tests/DoSkewSymCPRotationTest.cpp',
 'share/flit/litmus-tests/FMACancel.cpp',
 'share/flit/litmus-tests/InliningProblem.cpp',
 'share/flit/litmus-tests/Kahan.h',
 'share/flit/litmus-tests/KahanSum.cpp',
 'share/flit/litmus-tests/Matrix.h',
 'share/flit/litmus-tests/Paranoia.cpp',
 'share/flit/litmus-tests/RandHelper.cpp',
 'share/flit/litmus-tests/RandHelper.h',
 'share/flit/litmus-tests/ReciprocalMath.cpp',
 'share/flit/litmus-tests/RotateAndUnrotate.cpp',
 'share/flit/litmus-tests/RotateFullCircle.cpp',
 'share/flit/litmus-tests/Shewchuk.h',
 'share/flit/litmus-tests/ShewchukSum.cpp',
 'share/flit/litmus-tests/SimpleCHull.cpp',
 'share/flit/litmus-tests/SinInt.cpp',
 'share/flit/litmus-tests/TrianglePHeron.cpp',
 'share/flit/litmus-tests/TrianglePSylv.cpp',
 'share/flit/litmus-tests/Vector.h',
 'share/flit/litmus-tests/langois.cpp',
 'share/flit/litmus-tests/simple_convex_hull.cpp',
 'share/flit/litmus-tests/simple_convex_hull.h',
 'share/flit/litmus-tests/tinys.cpp',
 'share/flit/scripts',
 'share/flit/scripts/README.md',
 'share/flit/scripts/experimental',
 'share/flit/scripts/experimental/flit_ninja.py',
 'share/flit/scripts/experimental/ninja_syntax.py',
 'share/flit/scripts/flit.py',
 'share/flit/scripts/flit_bisect.py',
 'share/flit/scripts/flit_disguise.py',
 'share/flit/scripts/flit_experimental.py',
 'share/flit/scripts/flit_import.py',
 'share/flit/scripts/flit_init.py',
 'share/flit/scripts/flit_make.py',
 'share/flit/scripts/flit_update.py',
 'share/flit/scripts/flitconfig.py',
 'share/flit/scripts/flitelf.py',
 'share/flit/scripts/flitutil.py',
 'share/flit/src',
 'share/flit/src/ALL-FLIT.cpp',
 'share/flit/src/FlitCsv.cpp',
 'share/flit/src/InfoStream.cpp',
 'share/flit/src/TestBase.cpp',
 'share/flit/src/Variant.cpp',
 'share/flit/src/flit.cpp',
 'share/flit/src/flitHelpers.cpp',
 'share/flit/src/fsutil.cpp',
 'share/flit/src/subprocess.cpp',
 'share/flit/src/timeFunction.cpp',
 'share/licenses',
 'share/licenses/flit',
 'share/licenses/flit/LICENSE']

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
