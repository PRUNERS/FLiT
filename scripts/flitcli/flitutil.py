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
Utility functions shared between multiple flit subcommands.
'''

import flitconfig as conf

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import copy
import logging
import logging.config
import os
import socket
import sqlite3
import subprocess as subp
import sys
import tempfile
import json
import time

# cached values
_default_toml = None
_default_toml_string = None
SUPPORTED_COMPILER_TYPES = ('clang', 'gcc', 'intel')

def get_default_toml_string():
    '''
    Gets the default toml configuration file for FLiT and returns the string.
    '''
    global _default_toml_string
    if _default_toml_string is None:
        _default_toml_string = process_in_string(
            os.path.join(conf.config_dir, 'flit-default.toml.in'),
            {
                'flit_path': os.path.join(conf.script_dir, 'flit.py'),
                'config_dir': conf.config_dir,
                'flit_version': conf.version,
            })
    return _default_toml_string

def get_default_toml():
    '''
    Gets the default toml configuration file for FLIT and returns the
    configuration object.
    '''
    import toml
    global _default_toml
    if _default_toml is None:
        _default_toml = toml.loads(get_default_toml_string())
    return _default_toml

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
        for x in vals:
            fill_defaults(x, defaults[0])
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
    BEFORE = 0
    IN_LICENSE = 1
    AFTER = 2

    state = BEFORE
    filtered = []
    for line in lines:
        # state transitions
        if state == BEFORE and '-- LICENSE BEGIN --' in line:
            state = IN_LICENSE
            continue
        if state == IN_LICENSE and '-- LICENSE END --' in line:
            state = AFTER
            continue

        if state != IN_LICENSE:
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
    from os.path import isfile, getsize

    if not os.path.isfile(filename):
        return False

    # SQLite database file header is 100 bytes
    if os.path.getsize(filename) < 100:
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[:16] == b'SQLite format 3\000'

def extract_make_var(var, makefile='Makefile', directory='.'):
    '''
    Extracts the value of a particular variable within a particular Makefile.

    @param var: the name of the variable in the Makefile
    @param makefile: path to the Makefile (either absolute or relative to the
        given directory)
    @param directory: directory where the Makefile would be executed

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

    Make sure it works with names relative to the given directory

    Create a temporary directory
    >>> import tempfile
    >>> temporary_directory = tempfile.mkdtemp()

    Given a directory and an absolute path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('A  = hi there $(C)', file=makefile)
    ...     print('B  = my friend', file=makefile)
    ...     print('C := mike', file=makefile)
    ...     makefile.flush()
    ...     extract_make_var('A', makefile=makefilepath,
    ...                      directory=temporary_directory)
    ['hi', 'there', 'mike']

    Given a directory and a relative path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('A  = hi there $(C)', file=makefile)
    ...     print('B  = my friend', file=makefile)
    ...     print('C := mike', file=makefile)
    ...     makefile.flush()
    ...     extract_make_var('C', makefile=os.path.basename(makefilepath),
    ...                      directory=temporary_directory)
    ['mike']

    Delete the temporary directory
    >>> import shutil
    >>> shutil.rmtree(temporary_directory)
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

def extract_make_vars(makefile='Makefile', directory='.'):
    '''
    Extracts all GNU Make variables from the given Makefile, except for those
    that are built-in.  It is returned as a dictionary of
      {'var': ['val', ...]}

    @param makefile: path to the Makefile (absolute or relative to the given
        directory)
    @param directory: directory to use as the current directory where the
        Makefile would be run

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

    What if the file does not exist?  It throws an exception:

    >>> extract_make_var('A', 'file-should-not-exist.mk') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    subprocess.CalledProcessError: Command ... returned non-zero exit status 2.

    Make sure it works with names relative to the given directory

    Create a temporary directory
    >>> import tempfile
    >>> temporary_directory = tempfile.mkdtemp()

    Given a directory and an absolute path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('A  = hi there $(C)', file=makefile)
    ...     print('B  = my friend', file=makefile)
    ...     print('C := mike', file=makefile)
    ...     makefile.flush()
    ...     allvars = extract_make_vars(makefile=makefilepath,
    ...                                 directory=temporary_directory)
    >>> allvars['A']
    ['hi', 'there', 'mike']
    >>> allvars['B']
    ['my', 'friend']
    >>> allvars['C']
    ['mike']

    Given a directory and a relative path to a Makefile
    >>> makefilepath = os.path.join(temporary_directory, 'my-makefile.mk')
    >>> with open(makefilepath, 'w') as makefile:
    ...     print('A  = hi there $(C)', file=makefile)
    ...     print('B  = my friend', file=makefile)
    ...     print('C := mike', file=makefile)
    ...     makefile.flush()
    ...     allvars = extract_make_vars(makefile=os.path.basename(makefilepath),
    ...                                 directory=temporary_directory)
    >>> allvars['A']
    ['hi', 'there', 'mike']
    >>> allvars['B']
    ['my', 'friend']
    >>> allvars['C']
    ['mike']

    Delete the temporary directory
    >>> import shutil
    >>> shutil.rmtree(temporary_directory)
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

@contextmanager
def pushd(directory):
    '''
    Changes to a given directory using a with statement.  At the end of the
    with statement, changes back to the previous directory.

    >>> original_dir = os.path.abspath(os.curdir)
    >>> with tempdir() as new_dir:
    ...     temporary_directory = new_dir
    ...     with pushd(new_dir):
    ...         pushed_dir = os.path.abspath(os.curdir)
    ...     popped_dir = os.path.abspath(os.curdir)

    >>> temporary_directory == pushed_dir
    True
    >>> original_dir == popped_dir
    True
    >>> temporary_directory == popped_dir
    False
    '''
    original_dir = os.path.abspath(os.curdir)
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_dir)

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
    >>> os.path.isdir(temporary_directory)
    False
    >>> os.path.exists(temporary_directory)
    False

    Test that an exception is not thrown if it was already deleted
    >>> import shutil
    >>> with tempdir() as new_dir:
    ...     shutil.rmtree(new_dir)

    Test that the directory is still deleted even if an exception is thrown
    within the with statement.
    >>> try:
    ...     with tempdir() as new_dir:
    ...         temporary_directory = new_dir
    ...         raise RuntimeError()
    ... except RuntimeError:
    ...     pass
    >>> os.path.isdir(temporary_directory)
    False
    '''
    import tempfile
    import shutil
    new_dir = tempfile.mkdtemp(*args, **kwargs)
    try:
        yield new_dir
    finally:
        try:
            shutil.rmtree(new_dir)
        except FileNotFoundError:
            pass

def check_output(*args, **kwargs):
    '''
    Wrapper around subprocess.check_output() that returns a str object and
    suppresses standard error

    >>> check_output(['echo', 'hello there'])
    'hello there\\n'

    Output to standard error will be suppressed
    >>> check_output(['python', '-c', 'import sys; print("hi", file=sys.stderr)'])
    ''
    '''
    output = subp.check_output(stderr=subp.DEVNULL, *args, **kwargs)
    return output.decode(encoding='utf-8')


def get_event_log(event_name, message=''):
    '''
    Creates a JSON string for writing to event logs.
    '''

    event = {
        'date': str(datetime.utcnow()), 
        'time': time.perf_counter(),
        'type': event_name,
        'message': message
    }

    return json.dumps(event)


def write_log(events, path, filename, config=None):
    '''
    Utility for creating and writing FLiT event information to log files.
    
    Parameters:
    filename: Desired log file name. If file exists, data will be appended, 
      else file is created. It is left to the user to handle access conflicts.
    config: A dictionary in logging.dictConfig() format for configuring logger
      manually.
  
    -- Tests -- 
    Check that event_logs directory is created if it does not yet exist.
  
    Check that log file is generated if none exists
    >>> import os
    >>> 
    >>> os.path.exists('1.log')
    False
    >>> write_log('1.log')
    >>> os.path.exists('1.log')
    True
  
    Check that data is written to existing log file.
    >>> 
    '''
 
    # Create directory for logs, if necessary 
    Path(path).mkdir(parents=True, exist_ok=True)

    logfile = os.path.join(path, filename)

    if config is None:
        config = {
            'version': 1,
            'formatters': {
                'messageOnly': {
                    'class': 'logging.Formatter',
                    'format': '%(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': logfile,
                    'mode': 'w',
                    'formatter': 'messageOnly',
                }
            },
            'root': {
                'level': 'DEBUG',
                'handlers': ['file']
            },
        }
  
    logging.config.dictConfig(config)
    logger = logging.getLogger()    

    for event in events:
        logger.info(event)  
  

def parse_logs(directory):
    '''
    Utility for parsing multiple log files.
  
    
    '''
  
    pass
