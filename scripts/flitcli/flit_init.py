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

'Implements the init subcommand'

import argparse
import os
import shutil
import socket
import sys

import flitconfig as conf
import flitutil
import flit_update

brief_description = 'Initializes a flit test directory for use'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
        prog=prog,
        description='''
            Initializes a flit test directory for use.  It will initialize
            the directory by copying the default configuration file into
            the given directory.  If a configuration file already exists,
            this command does nothing.  The config file is called
            flit-config.toml.
            ''',
        )
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite init files if they are already there')
    parser.add_argument('-L', '--litmus-tests', action='store_true',
                        help='Copy over litmus tests too')
    args = parser.parse_args(arguments)

    os.makedirs(args.directory, exist_ok=True)

    # write flit-config.toml
    flit_config_dest = os.path.join(args.directory, 'flit-config.toml')
    print('Creating {0}'.format(flit_config_dest))
    with open(flit_config_dest, 'w') as fout:
        fout.write(flitutil.get_default_toml_string())

    def copy_files(dest_to_src, remove_license=True):
        '''
        @param dest_to_src: dictionary of dest -> src for copies
        @param remove_license: (default True) True means remove the license
            declaration at the top of each file copied.
        @return None
        '''
        for dest, src in sorted(dest_to_src.items()):
            realdest = os.path.join(args.directory, dest)
            print('Creating {0}'.format(realdest))
            if not args.overwrite and os.path.exists(realdest):
                print('Warning: {0} already exists, not overwriting'.format(realdest),
                      file=sys.stderr)
                continue
            os.makedirs(os.path.dirname(os.path.realpath(realdest)), exist_ok=True)
            if remove_license:
                with open(src, 'r') as fin:
                    with open(realdest, 'w') as fout:
                        fout.writelines(flitutil.remove_license_lines(fin))
            else:
                shutil.copy(src, realdest)

    # Copy the remaining files over
    to_copy = {
        'custom.mk': os.path.join(conf.data_dir, 'custom.mk'),
        'main.cpp': os.path.join(conf.data_dir, 'main.cpp'),
        'tests/Empty.cpp': os.path.join(conf.data_dir, 'tests/Empty.cpp'),
        }

    copy_files(to_copy, remove_license=True)

    # Add litmus tests too
    if args.litmus_tests:
        litmus_to_copy = {}
        for srcfile in os.listdir(conf.litmus_test_dir):
            if os.path.splitext(srcfile)[1] in ('.cpp', '.h'):
                srcpath = os.path.join(conf.litmus_test_dir, srcfile)
                litmus_to_copy[os.path.join('tests', srcfile)] = srcpath
        copy_files(litmus_to_copy)

    flit_update.main(['--directory', args.directory])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
