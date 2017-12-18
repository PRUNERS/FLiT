'''
Tests FLiT's capabilities to create a functioning empty project

The tests are below using doctest

Let's now make a temporary directory and test that flit init populates the
correct files
>>> import glob
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

Now we want to specifically test the Makefile and its use of flit-config.toml
and custom.mk in the subsequent tests.

'''

# Test setup before the docstring is run.
import doctest
import os
import sys
import subprocess as subp
from contextlib import contextmanager

sys.path.append('..')
import test_harness as th



if __name__ == '__main__':
    doctest.testmod()
