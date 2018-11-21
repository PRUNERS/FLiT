# Much of this is copied from the examples given in
#   https://github.com/eliben/pyelftools.git

from collections import namedtuple
import subprocess as subp
import sys
import os

from elftools.common.py3compat import bytes2str
from elftools.dwarf.descriptions import describe_form_class
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

SymbolTuple = namedtuple('SymbolTuple', 'src, symbol, demangled, fname, lineno')
SymbolTuple.__doc__ = '''
Tuple containing information about the symbols in a file.  Has the following
attributes:
    src:        source file that was compiled
    symbol:     mangled symbol in the compiled version
    demangled:  demangled version of symbol
    fname:      filename where the symbol is actually defined.  This usually
                will be equal to src, but may not be in some situations.
    lineno:     line number of definition within fname.
'''

def extract_symbols(objfile, srcfile):
    '''
    Extracts symbols for the given object file.

    @param objfile: (str) path to object file

    @return two lists of SymbolTuple objects (funcsyms, remaining).
        The first is the list of exported functions that are strong symbols and
        have a filename and line number where they are defined.  The second is
        all remaining symbols that are strong, exported, and defined.
    '''
    funcsym_tuples = []
    remainingsym_tuples = []

    with open(objfile, 'rb') as fin:
        elffile = ELFFile(fin)

        #symtabs = [elffile.get_section_by_name('.symtab')]
        symtabs = [x for x in elffile.iter_sections()
                   if isinstance(x, SymbolTableSection)]
        if len(symtabs) == 0:
            raise RuntimeError('Object file {} does not have a symbol table'
                               .format(objfile))

        # get globally exported defined symbols
        syms = [sym for symtab in symtabs
                    for sym in symtab.iter_symbols()
                    if _is_symbol(sym) and
                       _is_extern(sym) and
                       _is_strong(sym) and
                       _is_defined(sym)]

        # split symbols into functions and non-functions
        fsyms = [sym for sym in syms if _is_func(sym)] # functions
        rsyms = list(set(syms).difference(fsyms))      # remaining

        # TODO: find filename and line numbers for each relevant func symbol
        locs = _locate_symbols(elffile, fsyms)

        # TODO: demangle all symbols
        p = subp.Popen(['c++filt'], stdin=subp.PIPE, stdout=subp.PIPE)
        out, _ = p.communicate('\n'.join([sym.name for sym in fsyms]).encode())
        fdemangled = out.decode('utf8').splitlines()

        p = subp.Popen(['c++filt'], stdin=subp.PIPE, stdout=subp.PIPE)
        out, _ = p.communicate('\n'.join([sym.name for sym in rsyms]).encode())
        rdemangled = out.decode('utf8').splitlines()

        from pprint import pprint
        print('fdemangled = ', end='')
        pprint(fdemangled)

        assert len(fdemangled) == len(fsyms)
        funcsym_tuples = [SymbolTuple(srcfile, fsyms[i].name, fdemangled[i],
                                      locs[i][0], locs[i][1])
                          for i in range(len(fsyms))]
        remaining_tuples = [SymbolTuple(srcfile, rsyms[i].name, rdemangled[i],
                                         None, None)
                            for i in range(len(rsyms))]

    return funcsym_tuples, remaining_tuples

#    # use nm and objdump to get the binary information we need
#    symbol_strings = subp.check_output([
#        'nm',
#        '--extern-only',
#        '--defined-only',
#        '--line-numbers',
#        objfile,
#        ]).decode('utf-8').splitlines()
#    demangled_symbol_strings = subp.check_output([
#        'nm',
#        '--extern-only',
#        '--defined-only',
#        '--demangle',
#        objfile,
#        ]).decode('utf-8').splitlines()
#
#    print('extract_symbols({})'.format(fname))
#    if fname == 'tests/A.cpp':
#        from pprint import pprint
#        print('A.cpp: symbol strings:')
#        pprint(symbol_strings)
#        print('A.cpp: demangled symbol strings:')
#        pprint(demangled_symbol_strings)
#
#    # generate the symbol tuples
#    for symbol_string, demangled_string in zip(symbol_strings,
#                                               demangled_symbol_strings):
#        symbol_type, symbol = symbol_string.split(maxsplit=2)[1:]
#        demangled = demangled_string.split(maxsplit=2)[2]
#
#        if symbol_type == 'W':  # skip weak symbols
#            continue
#
#        if '\t' in symbol:  # if filename and linenumber are specified
#            symbol, definition = symbol.split('\t', maxsplit=1)
#            deffile, defline = definition.split(':')
#            defline = int(defline)
#            symtuple = SymbolTuple(fname, symbol, demangled, deffile, defline))
#            if symbol_type == 'T':
#                funcsym_tuples.append(symtuple)
#            else:
#                remainingsym_tuples.append(symtuple)
#        else:
#            remainingsym_tuples.append(
#                SymbolTuple(fname, symbol, demangled, None, None))
#
#    _extract_symbols_memos[objfile] = (funcsym_tuples, remainingsym_tuples)
#    return funcsym_tuples, remainingsym_tuples

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

def _locate_symbols(elffile, symbols):
    '''
    Locates the filename and line number of each symbol in the elf file.

    @param elffile: (elf.elffile.ELFFile) The top-level elf file
    @param symbols: (list(elf.sections.Symbol)) symbols to locate

    @return list(tuple(filename, lineno)) in the order of the given symbols

    If the file does not have DWARF info or a symbol is not found, an exception
    is raised
    '''
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
    '''
    # generate the table
    table = []
    for cu in dwarfinfo.iter_CUs():
        lineprog = dwarfinfo.line_program_for_CU(cu)
        prevstate = None
        for entry in lineprog.get_entries():
            # We're interested in those entries where a new state is assigned
            if entry.state is None or entry.state.end_sequence:
                continue
            # Looking for a range of addresses in two consecutive states that
            # contain a required address.
            if prevstate is not None:
                print(lineprog)
                filename = lineprog['file_entry'][prevstate.file - 1].name
                dirno = lineprog['file_entry'][prevstate.file - 1].dir_index
                filepath = os.path.join(lineprog['include_directory'][dirno - 1], filename)
                line = prevstate.line
                fromaddr = prevstate.address
                toaddr = max(fromaddr, entry.state.address)
                table.append((filepath, line, fromaddr, toaddr))
            prevstate = entry.state

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
