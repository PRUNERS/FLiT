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
Utility functions shared between multiple flit subcommands.
'''

import copy
import logging
import os
import socket
import sqlite3
import subprocess as subp
import sys
import tempfile

import flitconfig as conf

SUPPORTED_COMPILER_TYPES = ('clang', 'gcc', 'intel')

class Memoizer:
    '''
    A memoization decorator generator class.

    Example usage:
    >>> @Memoizer()
    ... def myfunction(i, j, k):
    ...     print('myfunction()')
    ...     return i
    >>> myfunction(1, 2, 3)
    myfunction()
    1
    >>> myfunction(1, 2, 3)
    1
    >>> myfunction(2, 3, 4)
    myfunction()
    2
    >>> myfunction(1, 2, 3)
    1
    >>> myfunction(2, 3, 4)
    2
    >>> myfunction(1, 3, 4)
    myfunction()
    1

    >>> @Memoizer(key=lambda i, j, k: i)
    ... def myfunction2(i, j, k):
    ...     print('myfunction2()')
    ...     return i
    >>> myfunction2(1, 2, 3)
    myfunction2()
    1
    >>> myfunction2(1, 3, 4)
    1
    >>> myfunction2(1, 2, 3)
    1
    >>> myfunction2(2, 3, 4)
    myfunction2()
    2
    >>> myfunction2(1, 2, 3)
    1
    >>> myfunction2(2, 3, 4)
    2
    >>> myfunction2(1, 3, 4)
    1
    '''

    def __init__(self, key=lambda *args, **kwargs: str(args) + str(kwargs)):
        self.key = key
        self.cache = {}

    def __call__(self, func):
        'Performs the decoration on func, using the given key function'
        def helper(*args, **kwargs):
            mykey = self.key(*args, **kwargs)
            if mykey not in self.cache:
                self.cache[mykey] = func(*args, **kwargs)
            return self.cache[mykey]
        # preserve the docstring (also for doctest to work)
        helper.__doc__ = func.__doc__
        return helper

    def __repr__(self):
        return repr(self.key)

def _cache_result(func):
    '''
    Decorator to decorate a function func (with no arguments) and cache the
    result after the first call to func.

    @param func (func): function taking no parameters and returning a value
    @return (func) a function that caches the result and only calls func() the
        first time (or more if func() previously returned None).

    >>> @_cache_result
    ... def hello():
    ...     'print hi there'
    ...     print('called hello()')
    ...     return 'hello'
    >>> hello()
    called hello()
    'hello'
    >>> hello()
    'hello'
    >>> hello()
    'hello'
    '''
    def helper():
        if not hasattr(helper, 'cache'):
            helper.cache = func()
        return helper.cache
    # preserve the docstring (also for doctest to work)
    helper.__doc__ = func.__doc__
    return helper

@_cache_result
def get_default_toml_string():
    '''
    Gets the default toml configuration file for FLiT and returns the string.
    '''
    return process_in_string(
        os.path.join(conf.config_dir, 'flit-default.toml.in'),
        {
            'flit_path': os.path.join(conf.script_dir, 'flit.py'),
            'config_dir': conf.config_dir,
            'hostname': socket.gethostname(),
            'flit_version': conf.version,
        })

@_cache_result
def get_default_toml():
    '''
    Gets the default toml configuration file for FLIT and returns the
    configuration object.
    '''
    import toml
    return toml.loads(get_default_toml_string())

def fill_defaults(vals, defaults=None):
    '''
    Given two combinations of dictionaries and lists (such as something
    generated from a json file or a toml file), enforce the defaults where the
    vals has missing values.

    - For dictionaries, missing keys will be populated with default values
    - For lists, this will recursively fill the defaults on each list item with
      the first list item in defaults (all other list items in defaults are
      ignored)

    Modifies vals and also returns the vals dictionary.

    If defaults is None, then the dictionary returned from get_default_toml()
    will be used.

    >>> fill_defaults({'a': 1}, {})
    {'a': 1}

    >>> fill_defaults({}, {'a': 1})
    {'a': 1}

    >>> fill_defaults({'a': 1}, {'a': 2})
    {'a': 1}

    >>> fill_defaults({'a': 2}, {'a': 1, 'b': 3})
    {'a': 2, 'b': 3}

    >>> fill_defaults([{}, {'a': 1}], [{'a': 2, 'b': 3}])
    [{'a': 2, 'b': 3}, {'a': 1, 'b': 3}]
    '''
    if defaults is None:
        defaults = get_default_toml()
    if isinstance(vals, dict):
        assert isinstance(defaults, dict)
        for key in defaults:
            if key not in vals:
                vals[key] = copy.deepcopy(defaults[key])
            else:
                fill_defaults(vals[key], defaults[key])
    elif isinstance(vals, list):
        assert isinstance(defaults, list)
        for value in vals:
            fill_defaults(value, defaults[0])
    return vals

def load_projconf(directory='.'):
    '''
    Loads and returns the project configuration found in the given tomlfile.
    This function checks for validity of that tomlfile and fills it with
    default values.

    @param directory: directory containing 'flit-config.toml'.

    @return project configuration as a struct of dicts and lists depending on
    the structure of the given tomlfile.
    '''
    import toml
    tomlfile = os.path.join(directory, 'flit-config.toml')
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        raise

    defaults = get_default_toml()

    if 'compiler' in projconf:
        assert isinstance(projconf['compiler'], list), \
            'flit-config.toml improperly configured, ' \
            'needs [[compiler]] section'

        default_type_map = {c['type']: c for c in defaults['compiler']}
        type_map = {} # type -> compiler
        name_map = {} # name -> compiler
        for compiler in projconf['compiler']:

            # make sure each compiler has a name, type, and binary
            for field in ('name', 'type', 'binary'):
                assert field in compiler, \
                    'flit-config.toml: compiler "{0}"'.format(compiler) + \
                    ' is missing the "{0}" field'.format(field)

            # check that the type is valid
            assert compiler['type'] in SUPPORTED_COMPILER_TYPES, \
                'flit-config.toml: unsupported compiler type "{0}"' \
                .format(compiler['type'])

            # check that we only have one of each type specified
            assert compiler['type'] not in type_map, \
                'flit-config.toml: cannot have multiple compilers of the ' \
                'same type ({0})'.format(compiler['type'])
            type_map[compiler['type']] = compiler

            # check that we only have one of each name specified
            assert compiler['name'] not in name_map, \
                'flit-config.toml: cannot have multiple compilers of the ' \
                'same name ({0})'.format(compiler['name'])
            name_map[compiler['name']] = compiler

            # if optimization_levels or switches_list are missing for any
            # compiler, put in the default flags for that compiler
            default = default_type_map[compiler['type']]
            for field in ('optimization_levels', 'switches_list'):
                if field not in compiler:
                    compiler[field] = default[field]

    # Fill in the rest of the default values
    fill_defaults(projconf, defaults)

    return projconf

def process_in_string(infile, vals, remove_license=True):
    '''
    Process a file such as 'Makefile.in' where there are variables to
    replace.  Returns a string with the replacements instead of outputting to a
    file.

    @param infile: input file.  Usually ends in ".in"
    @param vals: dictionary of key -> val where we search and replace {key}
        with val everywhere in the infile.
    @param remove_license: (default True) True means remove the License
        declaration at the top of the file that has "-- LICENSE BEGIN --" at
        the beginning and "-- LICENSE END --" at the end.  All lines between
        including those lines will be removed.  False means ignore the license
        section.  If the license section is not there, then this will have no
        effect (except for a slight slowdown for searching)
    @return processed string
    '''
    with open(infile, 'r') as fin:
        if remove_license:
            fin_content = ''.join(remove_license_lines(fin))
        else:
            fin_content = fin.read()
    return fin_content.format(**vals)

def process_in_file(infile, dest, vals, overwrite=False, remove_license=True):
    '''
    Process a file such as 'Makefile.in' where there are variables to
    replace.

    @param infile: input file.  Usually ends in ".in"
    @param dest: destination file.  If overwrite is False, then destination
        shouldn't exist, otherwise a warning is printed and nothing is
        done.
    @param vals: dictionary of key -> val where we search and replace {key}
        with val everywhere in the infile.
    @param overwrite: (default False) True means overwrite the destination.  If
        False, then a warning will be printed to the console and this function
        will return without doing anything.
    @param remove_license: (default True) True means remove the License
        declaration at the top of the file that has "-- LICENSE BEGIN --" at
        the beginning and "-- LICENSE END --" at the end.  All lines between
        including those lines will be removed.  False means ignore the license
        section.  If the license section is not there, then this will have no
        effect (except for a slight slowdown for searching)
    @return None
    '''
    if not overwrite and os.path.exists(dest):
        print('Warning: {0} already exists, not overwriting'.format(dest),
              file=sys.stderr)
        return
    content = process_in_string(infile, vals, remove_license=remove_license)
    with open(dest, 'w') as fout:
        fout.write(content)

def remove_license_lines(lines):
    '''
    Removes the license lines from the iterable of lines.  The license lines
    section will start with a line containing "-- LICENSE BEGIN --" and will
    end with a line containing "-- LICENSE END --".  All lines between the two
    including those two lines will be removed.

    @param lines: the iterable of lines to filter
    @return lines

    >>> remove_license_lines([
    ...     '-- something here --',
    ...     'bla bla bla -- LICENSE BEGIN --',
    ...     'my name is part of the license!',
    ...     '# -- LICENSE END -- **/',
    ...     'hello',
    ...     ])
    ['-- something here --', 'hello']

    >>> remove_license_lines([])
    []

    >>> remove_license_lines(['-- LICENSE BEGIN --', 'bla bla bla'])
    []

    >>> remove_license_lines(['bla bla bla', '-- LICENSE END --'])
    ['bla bla bla', '-- LICENSE END --']

    >>> lines = ['hello', '-- LICENSE BEGIN --', '-- LICENSE END --']
    >>> remove_license_lines(lines)
    ['hello']
    '''
    # states
    state_before = 0
    state_in_license = 1
    state_after = 2

    state = state_before
    filtered = []
    for line in lines:
        # state transitions
        if state == state_before and '-- LICENSE BEGIN --' in line:
            state = state_in_license
            continue
        if state == state_in_license and '-- LICENSE END --' in line:
            state = state_after
            continue

        if state != state_in_license:
            filtered.append(line)

    return filtered

def sqlite_open(filepath):
    '''
    Opens and returns an sqlite database cursor object.  If the database does
    not exist, it will be created.
    '''
    # Using detect_types allows us to insert datetime objects
    connection = sqlite3.connect(filepath,
                                 detect_types=sqlite3.PARSE_DECLTYPES)

    # Use the dict factory so that queries return dictionary-like objects
    connection.row_factory = sqlite3.Row

    # Create the tables if they do not exist.  Also has other setup.
    table_file = os.path.join(conf.data_dir, 'db', 'tables-sqlite.sql')
    with open(table_file, 'r') as table_sql:
        connection.executescript(table_sql.read())
    connection.commit()

    return connection

def is_sqlite(filename):
    'Returns true if the file is likely an sqlite file.'
    if not os.path.isfile(filename):
        return False

    # SQLite database file header is 100 bytes
    if os.path.getsize(filename) < 100:
        return False

    with open(filename, 'rb') as file_in:
        header = file_in.read(100)

    return header[:16] == b'SQLite format 3\000'

def extract_make_var(var, makefile='Makefile', directory='.'):
    '''
    Extracts the value of a particular variable within a particular Makefile.

    How it works with a valid file:

    >>> from tempfile import NamedTemporaryFile as NTF
    >>> with NTF(mode='w+') as fout:
    ...     print('A    := hello there sweetheart\\n', file=fout, flush=True)
    ...     A = extract_make_var('A', fout.name)
    >>> A
    ['hello', 'there', 'sweetheart']

    If the variable is undefined, then simply an empty list is returned.

    >>> with NTF() as fout: extract_make_var('A', fout.name)
    []

    What if the file does not exist?  It throws an exception:

    >>> extract_make_var('A', 'file-should-not-exist.mk') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    subprocess.CalledProcessError: Command ... returned non-zero exit status 2.
    '''
    with tempfile.NamedTemporaryFile(mode='w+') as fout:
        print('print-%:\n'
              "\t@echo '$*=$($*)'\n", file=fout, flush=True)
        output = subp.check_output(
            ['make', '-f', makefile, '-f', fout.name, 'print-' + var,
             '--directory', directory, '--no-print-directory'],
            stderr=subp.STDOUT)
    output = output.strip().decode('utf-8')
    var_values = output.split('=', maxsplit=1)[1].split()
    return var_values

def _extract_make_vars_memoizer_key(makefile='Makefile', directory='.'):
    'Key function for the memoizer of extract_make_vars()'
    return (directory, makefile)

@Memoizer(key=_extract_make_vars_memoizer_key)
def extract_make_vars(makefile='Makefile', directory='.'):
    '''
    Extracts all GNU Make variables from the given Makefile, except for those
    that are built-in.  It is returned as a dictionary of
      {'var': ['val', ...]}

    @note, all variables are returned, including internal Makefile
    variables.

    >>> from tempfile import NamedTemporaryFile as NTF
    >>> with NTF(mode='w+') as fout:
    ...     print('A    := hello there sweetheart\\n'
    ...           'B     = \\n',
    ...           file=fout, flush=True)
    ...     allvars = extract_make_vars(fout.name)

    >>> allvars['A']
    ['hello', 'there', 'sweetheart']
    >>> allvars['B']
    []

    >>> allvars == extract_make_vars(fout.name)
    True
    >>> allvars == extract_make_vars(fout.name, '.')
    True
    >>> allvars == extract_make_vars(makefile=fout.name)
    True
    >>> allvars == extract_make_vars(directory='.', makefile=fout.name)
    True

    What if the file does not exist?  It throws an exception:

    >>> extract_make_var('A', 'file-should-not-exist.mk') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    subprocess.CalledProcessError: Command ... returned non-zero exit status 2.
    '''
    with tempfile.NamedTemporaryFile(mode='w+') as fout:
        print('$(foreach v,$(.VARIABLES),$(info $(v)=$($(v))**==**))\n'
              '.PHONY: empty-target\n'
              'empty-target:\n'
              "\t@true\n", file=fout, flush=True)
        output = subp.check_output(
            ['make', '-f', makefile, '-f', fout.name, 'empty-target',
             '--directory', directory, '--no-print-directory'],
            stderr=subp.DEVNULL)
    lines = output.strip().decode('utf-8').splitlines()
    var_values = {}
    prevkey = None
    for line in lines:
        if prevkey is None:
            split = line.split('=', maxsplit=1)
            prevkey = split[0]
            values = split[1]
            var_values[prevkey] = []
        else:
            values = line

        key = prevkey

        if values.endswith('**==**'):
            values = values.replace('**==**', '')
            prevkey = None

        var_values[key].extend(values.split())

    return var_values

def printlog(message):
    '''
    Prints the message to stdout and then logs the message at the info level.
    It is expected that the logging module has already been configured using
    the root logger.

    >>> logger = logging.getLogger()
    >>> for handler in logger.handlers[:]:
    ...     logger.removeHandler(handler)
    >>> import sys
    >>> logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    >>> printlog('Danger Will Robinson!')
    Danger Will Robinson!
    INFO:root:Danger Will Robinson!
    '''
    print(message)
    logging.info(message)
