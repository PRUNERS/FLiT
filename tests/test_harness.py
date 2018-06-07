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

    Test that the temporary directory exists during the block and is removed
    after
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

    Test that an exception is not thrown if it was already deleted
    >>> import shutil
    >>> with tempdir() as new_dir:
    ...     shutil.rmtree(new_dir)
    '''
    import tempfile
    import shutil
    new_dir = tempfile.mkdtemp(*args, **kwargs)
    yield new_dir
    try:
        shutil.rmtree(new_dir)
    except FileNotFoundError:
        pass

def touch(filename):
    '''
    Create an emtpy file if it does not exist, otherwise updates the
    modification time.
    '''
    from pathlib import Path
    Path(filename).touch()

flit = _path_import(_script_dir, 'flit')
config = _path_import(_script_dir, 'flitconfig')

# Remove the things that are no longer necessary
del contextmanager
