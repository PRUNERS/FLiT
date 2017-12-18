from contextlib import contextmanager
#from importlib.machinery import SourceFileLoader
import importlib

import glob
import os
import sys

_harness_dir = os.path.dirname(os.path.realpath(__file__))
_flit_dir = os.path.dirname(_harness_dir)
_script_dir = os.path.join(_flit_dir, 'scripts/flitcli')

@contextmanager
def _add_to_path(p):
    'Adds a path to sys.path in a context'
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path

def _path_import(module_dir, name):
    '''
    implementation taken from
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    '''
    with _add_to_path(module_dir):
        # This is the way to do it with python 3.5+
        #spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        #module = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(module)
        #return module
        # This is the way to do it with python 3.3+, but is depricated in 3.5+
        #return SourceFileLoader(name, fpath).load_module()
        # This is the way to do it with python 2.7+
        # This will still work until Python 4
        return importlib.import_module(name)

flit = _path_import(_script_dir, 'flit')
config = _path_import(_script_dir, 'flitconfig')

del os
del glob
