'''
Tests FLiT's generated Makefile and its ability to correctly do incremental
builds.

The tests are below using doctest

>>> import glob
>>> import os
>>> import subprocess as subp
>>> import re
>>> import time

Test that the dev target will be rebuilt if one of the files is updated.
We use 'make --touch' to simply touch files it would create and then we will
test that the correct files are updated.
>>> def compile_target(directory, target):
...     'Compiles the dev target using "make --touch" and returns the output'
...     output = subp.check_output(['make', '-C', directory, '--touch', target])
...     return output.decode('utf-8').strip()
>>> compile_dev = lambda x: compile_target(x, 'dev')
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     # fake the built elements -- don't actually build for time's sake
...     os.mkdir(os.path.join(temp_dir, 'obj'))
...     before_build = compile_dev(temp_dir)
...     # this next build should say nothing to do
...     after_build = compile_dev(temp_dir)
...     # update one file and recompile
...     with open(os.path.join(temp_dir, 'main.cpp'), 'a') as mainfile:
...         mainfile.write('#include "new_header.h"\\n')
...     with open(os.path.join(temp_dir, 'obj', 'main_dev.d'), 'w') as maindep:
...         maindep.write('obj/main_dev.o: main.cpp new_header.h\\n')
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_modify = compile_dev(temp_dir)
...     # touch the header file and make sure it recompiles again
...     time.sleep(0.01) # give some time before touching again
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_touch = compile_dev(temp_dir)
Creating ...

>>> def touched_files(outstring):
...     'Returns list of touched files in sorted order'
...     return sorted([x[6:] for x in outstring.splitlines()
...                    if x.startswith('touch ')])

Make sure all of the correct files were created with our build commands

>>> touched_files(before_build)
['devrun', 'obj/Empty_dev.o', 'obj/main_dev.o']

>>> touched_files(after_build)
[]

>>> touched_files(after_modify)
['devrun', 'obj/main_dev.o']

>>> touched_files(after_touch)
['devrun', 'obj/main_dev.o']

Now, let's test the same thing with the "gt" target

>>> compile_gt = lambda x: compile_target(x, 'gt')
>>> with th.tempdir() as temp_dir:
...     th.flit.main(['init', '-C', temp_dir]) # doctest:+ELLIPSIS
...     # fake the built elements -- don't actually build for time's sake
...     os.mkdir(os.path.join(temp_dir, 'obj'))
...     before_build = compile_gt(temp_dir)
...     # this next build should say nothing to do
...     after_build = compile_gt(temp_dir)
...     # update one file and recompile
...     with open(os.path.join(temp_dir, 'main.cpp'), 'a') as mainfile:
...         mainfile.write('#include "new_header.h"\\n')
...     with open(os.path.join(temp_dir, 'obj', 'main_gt.d'), 'w') as maindep:
...         maindep.write('obj/main_gt.o: main.cpp new_header.h\\n')
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_modify = compile_gt(temp_dir)
...     # touch the header file and make sure it recompiles again
...     time.sleep(0.01) # give some time before touching again
...     th.touch(os.path.join(temp_dir, 'new_header.h'))
...     after_touch = compile_gt(temp_dir)
Creating ...

Make sure all of the correct files were created with our build commands

>>> touched_files(before_build)
['gtrun', 'obj/Empty_gt.o', 'obj/main_gt.o']

>>> touched_files(after_build)
[]

>>> touched_files(after_modify)
['gtrun', 'obj/main_gt.o']

>>> touched_files(after_touch)
['gtrun', 'obj/main_gt.o']
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
