# Much of this is copied from the examples given in
#   https://github.com/eliben/pyelftools.git

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
Utility functions for dealing with ELF binary files.  This file requires the
pyelftools package to be installed (i.e. module elftools).
'''

from collections import namedtuple
import subprocess as subp
import os

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

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

def extract_symbols(objfile):
    '''
    Extracts symbols for the given object file.

    @param objfile: (str) path to object file

    @return two lists of SymbolTuple objects (funcsyms, remaining).
        The first is the list of exported functions that are strong symbols and
        have a filename and line number where they are defined.  The second is
        all remaining symbols that are strong, exported, and defined.
    '''
    with open(objfile, 'rb') as fin:
        elffile = ELFFile(fin)

        symtabs = [x for x in elffile.iter_sections()
                   if isinstance(x, SymbolTableSection)]
        if len(symtabs) == 0:
            raise RuntimeError('Object file {} does not have a symbol table'
                               .format(objfile))

        # get globally exported defined symbols
        syms = [sym for symtab in symtabs
                for sym in symtab.iter_symbols()
                if _is_symbol(sym)
                and _is_extern(sym)
                and _is_strong(sym)
                and _is_defined(sym)]

        # split symbols into functions and non-functions
        fsyms = [sym for sym in syms if _is_func(sym)] # functions
        rsyms = list(set(syms).difference(fsyms))      # remaining

        # find filename and line numbers for each relevant func symbol
        locs = _locate_symbols(elffile, fsyms)

        # demangle all symbols
        fdemangled = _demangle([sym.name for sym in fsyms])
        rdemangled = _demangle([sym.name for sym in rsyms])

        funcsym_tuples = [SymbolTuple(fsyms[i].name, fdemangled[i],
                                      locs[i][0], locs[i][1])
                          for i in range(len(fsyms))]
        remaining_tuples = [SymbolTuple(rsyms[i].name, rdemangled[i],
                                        None, None)
                            for i in range(len(rsyms))]

        return funcsym_tuples, remaining_tuples

def _symbols(symtab):
    'Returns all symbols from the given symbol table'
    return [sym for sym in symtab.iter_symbols() if _is_symbol(sym)]

def _is_symbol(sym):
    'Returns True if elf.sections.Symbol object is a symbol'
    return sym.name != '' and sym['st_info']['type'] != 'STT_FILE'

def _is_extern(sym):
    'Returns True if elf.sections.Symbol is an extern symbol'
    return sym['st_info']['bind'] != 'STB_LOCAL'

def _is_weak(sym):
    'Returns True if elf.sections.Symbol is a weak symbol'
    return sym['st_info']['bind'] == 'STB_WEAK'

def _is_strong(sym):
    'Returns True if elf.sections.Symbol is a strong symbol'
    return sym['st_info']['bind'] == 'STB_GLOBAL'

def _is_defined(sym):
    'Returns True if elf.sections.Symbol is defined'
    return sym['st_shndx'] != 'SHN_UNDEF'

def _is_func(sym):
    'Returns True if elf.sections.Symbol is a function'
    return sym['st_info']['type'] == 'STT_FUNC'

def _demangle(symbol_list):
    'Demangles each C++ name in the given list'
    proc = subp.Popen(['c++filt'], stdin=subp.PIPE, stdout=subp.PIPE)
    out, _ = proc.communicate('\n'.join(symbol_list).encode())
    demangled = out.decode('utf8').splitlines()
    assert len(demangled) == len(symbol_list)
    return demangled

def _locate_symbols(elffile, symbols):
    '''
    Locates the filename and line number of each symbol in the elf file.

    @param elffile: (elf.elffile.ELFFile) The top-level elf file
    @param symbols: (list(elf.sections.Symbol)) symbols to locate

    @return list(tuple(filename, lineno)) in the order of the given symbols

    If the file does not have DWARF info or a symbol is not found, an exception
    is raised

    Test that even without a proper elffile, if there are no symbols to match,
    then no error occurs and you can be on your merry way.
    >>> _locate_symbols(object(), [])
    []
    '''
    if len(symbols) == 0:
        return []

    if not elffile.has_dwarf_info():
        raise RuntimeError('Elf file has no DWARF info')

    dwarfinfo = elffile.get_dwarf_info()
    fltable = _gen_file_line_table(dwarfinfo)

    locations = []
    for sym in symbols:
        for fname, lineno, start, end in fltable:
            if start <= sym.entry['st_value'] < end:
                locations.append((fname.decode('utf8'), lineno))
                break
        else:
            locations.append((None, None))

    return locations

def _gen_file_line_table(dwarfinfo):
    '''
    Generates and returns a list of (filename, lineno, startaddr, endaddr).

    Tests that an empty dwarfinfo object will result in an empty return list
    >>> class FakeDwarf:
    ...     def __init__(self):
    ...         pass
    ...     def iter_CUs(self):
    ...         return []
    >>> _gen_file_line_table(FakeDwarf())
    []
    '''
    # generate the table
    table = []
    for unit in dwarfinfo.iter_CUs():
        lineprog = dwarfinfo.line_program_for_CU(unit)
        prevstate = None
        for entry in lineprog.get_entries():
            # We're interested in those entries where a new state is assigned
            if entry.state is None or entry.state.end_sequence:
                continue
            # Looking for a range of addresses in two consecutive states that
            # contain a required address.
            if prevstate is not None:
                filename = lineprog['file_entry'][prevstate.file - 1].name
                dirno = lineprog['file_entry'][prevstate.file - 1].dir_index
                filepath = os.path.join(
                        lineprog['include_directory'][dirno - 1], filename)
                line = prevstate.line
                fromaddr = prevstate.address
                toaddr = max(fromaddr, entry.state.address)
                table.append((filepath, line, fromaddr, toaddr))
            prevstate = entry.state

    # If there are no functions, then return an empty list
    if len(table) == 0:
        return []

    # consolidate the table
    consolidated = []
    prev = table[0]
    for entry in table[1:]:
        if prev[1] == entry[1] and prev[3] == entry[2]:
            prev = (prev[0], prev[1], prev[2], entry[3])
        else:
            consolidated.append(prev)
            prev = entry
    consolidated.append(prev)

    return consolidated
