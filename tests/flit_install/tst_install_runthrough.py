'''
Tests FLiT's capabilities to run simple commands on an installation

The tests are below using doctest

Let's now make a temporary directory and install there.  Here we are simply
testing that the following command complete without error.
>>> import glob
>>> import os
>>> import subprocess as subp
>>> with th.tempdir() as temp_dir:
...     _ = subp.check_call(['make', '-C', os.path.join(th.config.lib_dir, '..'),
...                          'install', 'PREFIX=' + temp_dir],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     flit = os.path.join(temp_dir, 'bin', 'flit')
...     _ = subp.check_call([flit, 'init', '-C', os.path.join(temp_dir, 'sandbox')],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     _ = subp.check_call(['mkdir', '-p', os.path.join(temp_dir, 'sandbox', 'obj'),
...                          os.path.join(temp_dir, 'sandbox', 'results')],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     _ = subp.check_call(['make', '-C', os.path.join(temp_dir, 'sandbox'),
...                          '--touch', 'run'],
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
...     os.chdir(os.path.join(temp_dir, 'sandbox'))
...     _ = subp.check_call([flit, 'import'] + glob.glob('results/*.csv'),
...                         stdout=subp.DEVNULL, stderr=subp.DEVNULL)
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
