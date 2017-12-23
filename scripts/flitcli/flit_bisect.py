'''
Implements the bisect subcommand, identifying the problematic subset of source
files that cause the variability.
'''

import flitutil as util

import toml

import argparse
import csv
import datetime
import os
import sqlite3
import sys

brief_description = 'Bisect compilation to identify problematic source code'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                Compiles the source code under both the ground-truth
                compilation and a given problematic compilation.  This tool
                then finds the minimal set of source files needed to be
                compiled under the problematic compilation flags so that the
                same answer is given.  This allows you to narrow down where the
                reproducibility problem lies.
                ''',
            )
    parser.add_argument('compilation',
                        help='''
                            The problematic compilation to use.  This should
                            specify the compiler, optimization level, and
                            switches (which can be empty).  An example value
                            for this option would be "gcc -O2
                            -funsafe-math-optimizations" or
                            "/opt/intel/bin/icpc -O1".  The value will be split
                            into three groups using space separators, the first
                            is the compiler, the second is the optimization
                            level, and the third (if present) is the switches.
                            You will likely want to have this argument in
                            quotes since it will contain spaces.
                            ''')
    args = parser.parse_args(arguments)

    tomlfile = 'flit-config.toml'
    try:
        projconf = toml.load(tomlfile)
    except FileNotFoundError:
        print('Error: {0} not found.  Run "flit init"'.format(tomlfile),
              file=sys.stderr)
        return 1

    # Split the compilation into the separate components
    compiler, optl, switches = args.compilation.strip().split(maxsplit=2)

    # TODO: see if the Makefile needs to be regenerated

    # get the list of source files from the Makefile
    sources = util.extract_make_var('SOURCE', 'Makefile')

    # TODO: determine if the problem is on the linker's side
    #         I'm not yet sure the best way to do that

    # TODO: Perform in parallel binary search from ground-truth and from
    #       problematic
    # - create extra Makefile
    # - run extra Makefile to determine if the problem is still there
    # - 

    # TODO: Use the custom comparison function in the test class when
    #       performing the binary search.

    # TODO: autogenerate Makefiles in the /tmp directly, preferrably with the
    #       tempfile module.


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
