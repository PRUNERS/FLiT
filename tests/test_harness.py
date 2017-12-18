'''
Sets up an environment for testing.  The tests are performed using doctest, so
simply write your tests that way.

This module provides the following things:
- tempdir():  a function for creating a temporary directory in a with statement
- flit: the main flit module.  Use the main() function with cli arguments
- config: the flitconfig module with paths
'''

from contextlib import contextmanager

import os

_harness_dir = os.path.dirname(os.path.realpath(__file__))
_flit_dir = os.path.dirname(_harness_dir)
_script_dir = os.path.join(_flit_dir, 'scripts/flitcli')

del os

@contextmanager
def _add_to_path(p):
    '''
    Adds a path to sys.path in a context

    >>> import sys
    >>> before_path = sys.path[:]
    >>> to_add = '/jbasdr/july'
    >>> print(to_add in sys.path)
    False
    >>> with _add_to_path(to_add):
    ...     print(to_add in sys.path)
    ...
    True
    >>> to_add in sys.path
    False
    >>> before_path == sys.path
    True
    '''
    import sys
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path

def _path_import(module_dir, name):
    '''
    Imports a named module from the specified directory and the specified
    module name from that directory.

    implementation taken from
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    >>> import tempfile
    >>> import os
    >>> import sys
    >>> before_path = sys.path[:]
    >>> with tempfile.NamedTemporaryFile(suffix='.py') as temporary:
    ...     _ = temporary.write(b'a = [5,4,3,2,1]\\n')
    ...     temporary.flush()
    ...     temporary_module = _path_import(os.path.dirname(temporary.name),
    ...                                     os.path.basename(temporary.name)[:-3])
    ...
    >>> temporary_module.a
    [5, 4, 3, 2, 1]

    This function is guaranteed to not have the system path modified afterwards
    >>> sys.path == before_path
    True
    '''
    with _add_to_path(module_dir):
        # This is the way to do it with python 3.5+
        #import importlib.util
        #spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        #module = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(module)
        #return module
        # This is the way to do it with python 3.3+, but is depricated in 3.5+
        #from importlib.machinery import SourceFileLoader
        #return SourceFileLoader(name, fpath).load_module()
        # This is the way to do it with python 2.7+
        # This will still work until Python 4
        import importlib
        return importlib.import_module(name)

@contextmanager
def tempdir(*args, **kwargs):
    '''
    Creates a temporary directory using tempfile.mkdtemp().  All arguments are
    passed there.  This function is to be used in a with statement.  At the end
    of the with statement, the temporary directory will be deleted with
    everything in it.

    >>> import os
    >>> temporary_directory = None
    >>> with tempdir() as new_dir:
    ...     temporary_directory = new_dir
    ...     print(os.path.isdir(temporary_directory))
    ...
    True
    >>> print(os.path.isdir(temporary_directory))
    False
    >>> print(os.path.exists(temporary_directory))
    False
    '''
    import tempfile
    import shutil
    new_dir = tempfile.mkdtemp(*args, **kwargs)
    yield new_dir
    shutil.rmtree(new_dir)

flit = _path_import(_script_dir, 'flit')
config = _path_import(_script_dir, 'flitconfig')

# Remove the things that are no longer necessary
del contextmanager
