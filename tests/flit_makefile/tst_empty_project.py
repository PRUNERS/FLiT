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

>>> import subprocess as subp
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     _ = subp.check_output(['make', '-C', temp_dir, 'dev', '-j8'],
...                           stderr=subp.STDOUT)
...     devrun_out = subp.check_output([os.path.join(temp_dir, 'devrun')],
...                                    stderr=subp.STDOUT)
...     print(devrun_out.decode('utf-8'), end='')
Creating ...
name,host,compiler,optl,switches,precision,score,score_d,resultfile,comparison,comparison_d,file,nanosec

TODO: let's test that the full set of tests would get created.
TODO: test the cuda generation,
TODO: test that the custom.mk stuff gets used
TODO: test CUDA_ONLY
TODO: test CLANG_ONLY
TODO: test targets dev, devcuda, gt, run, runbuild, clean, veryclean, distclean
TODO: test solution on a mac (with UNAME_S being Darwin)
'''

# Test setup before the docstring is run.
import sys
before_path = sys.path[:]
sys.path.append('..')
import test_harness as th
sys.path = before_path

if __name__ == '__main__':
    import doctest
    doctest.testmod()
