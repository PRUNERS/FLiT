'''
Tests FLiT's --version argument

The tests are below using doctest

Let's now make a temporary directory and test that flit init populates the
correct files

>>> import subprocess as subp
>>> import os

>>> actual = subp.check_output([os.path.join(th.config.script_dir, 'flit.py'),
...                             '--version'])
>>> actual = actual.decode('utf-8').strip()
>>> expected = 'flit version ' + th.config.version
>>> expected == actual
True

>>> actual = subp.check_output([os.path.join(th.config.script_dir, 'flit.py'), '-v'])
>>> actual = actual.decode('utf-8').strip()
>>> expected = 'flit version ' + th.config.version
>>> expected == actual
True

>>> actual = subp.check_output([os.path.join(th.config.script_dir, 'flit.py'),
...                             '--help'])
>>> actual = actual.decode('utf-8').strip()
>>> '-v, --version' in actual
True
>>> 'Print version and exit' in actual
True


Check that an installed flit will also be able to handle the call to --version

>>> with th.tempdir() as tmpdir:
...     _ = subp.check_output(['make', 'install', 'PREFIX=' + tmpdir, '-C',
...             os.path.dirname(th.config.lib_dir)])
...     actual = subp.check_output([os.path.join(tmpdir, 'bin', 'flit'), '-v'])
>>> actual = actual.decode('utf-8').strip()
>>> expected = 'flit version ' + th.config.version
>>> expected == actual
True
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
