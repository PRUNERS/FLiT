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
Utility functions for dealing with ELF binary files.  This file uses
alternative methods to do this functionality that does not require the
pyelftools package.  But this file gives the same public interface as
pyelftools so that it can be used as a replacement.

Instead, this package uses binutils through subprocesses.  The programs used
are "nm" and "c++filt" to perform the same functionality.
'''

from collections import namedtuple
import subprocess as subp
import os
import shutil

if not shutil.which('nm') or not shutil.which('c++filt'):
    raise ImportError('Cannot find binaries "nm" and "c++filt"')

SymbolTuple = namedtuple('SymbolTuple',
                         'symbol, demangled, fname, lineno')
SymbolTuple.__doc__ = '''
Tuple containing information about the symbols in a file.  Has the following
attributes:
    symbol:     mangled symbol in the compiled version
    demangled:  demangled version of symbol
    fname:      filename where the symbol is defined.
    lineno:     line number of definition within fname.
'''

def extract_symbols(objfile_or_list):
    '''
    Extracts symbols for the given object file.

    @param objfile_or_list: (str or list(str)) path to object file(s)

    @return two lists of SymbolTuple objects (funcsyms, remaining).
        The first is the list of exported functions that are strong symbols and
        have a filename and line number where they are defined.  The second is
        all remaining symbols that are strong, exported, and defined.
    '''
    funcsym_tuples = []
    remaining_tuples = []
    nm_args = [
        'nm',
        '--print-file-name',
        '--extern-only',
        '--defined-only',
        '--line-numbers',
        ]
    if isinstance(objfile_or_list, str):
        nm_args.append(objfile_or_list)
    else:
        nm_args.extend(objfile_or_list)
    symbol_strings = subp.check_output(nm_args).decode('utf-8').splitlines()

    symbols = []
    filenames = []
    linenumbers = []
    for symbol_string in symbol_strings:
        try:
            stype, symbol_plus_extra = symbol_string.split(maxsplit=2)[1:]
        except:
            import pdb; pdb.set_trace()
        if stype.lower() == 'w':
            symbol = symbol_plus_extra.split()[0]
            filename = None
            linenumber = None
        elif '\t' in symbol_plus_extra:
            symbol, definition = symbol_plus_extra.split('\t', maxsplit=1)
            filename, linenumber= definition.split(':')
            linenumber = int(linenumber)
        else:
            symbol = symbol_plus_extra
            filename = None
            linenumber = None
        symbols.append(symbol)
        filenames.append(filename)
        linenumbers.append(linenumber)

    demangled = _demangle(symbols)

    for sym, dem, fnam, line in zip(symbols, demangled, filenames, linenumbers):
        symbol_tuple = SymbolTuple(sym, dem, fnam, line)
        if fnam:
            funcsym_tuples.append(symbol_tuple)
        else:
            remaining_tuples.append(symbol_tuple)

    return funcsym_tuples, remaining_tuples

def _demangle(symbol_list):
    'Demangles each C++ name in the given list'
    proc = subp.Popen(['c++filt'], stdin=subp.PIPE, stdout=subp.PIPE)
    out, _ = proc.communicate('\n'.join(symbol_list).encode())
    demangled = out.decode('utf8').splitlines()
    assert len(demangled) == len(symbol_list)
    return demangled