#!/usr/bin/env python3
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
'Pretend to be clang 3.4, specifically checking for unsupported flags'

import sys

VERSION = '3.4.1'

def print_version():
    'Print fake version information'
    nodot_version = VERSION.replace('.', '')
    print('clang version {0} (tags/{1}/final)'.format(VERSION, nodot_version))
    print('Target: x86_64-pc-linux-gnu')
    print('Thread model: posix')
    print('InstalledDir: /usr/bin')

def main(arguments):
    'Main logic here'

    recognized_arguments = [
        '-fno-pie',
        '-std',
        '-g',
        '-o',
        '-nopie',
        '-fassociative-math',
        '-mavx',
        '-fexcess-precision',
        '-ffinite-math-only',
        '-mavx2',
        '-mfma',
        '-march',
        '-ffp-contract',
        '-ffloat-store',
        '-fmerge-all-constants',
        '-fno-trapping-math',
        '-freciprocal-math',
        '-frounding-math',
        '-fsignaling-nans',
        '-fsingle-precision-constant',
        '-mfpmath',
        '-mtune',
        '-funsafe-math-optimizations',
        '-MMD',
        '-MP',
        '-MF',
        '--gcc-toolchain',
        '-c'
        ]

    recognized_beginnings = [
        '-W',
        '-D',
        '-I',
        '-l',
        '-L',
        '-O',
        ]

    if '--version' in arguments:
        print_version()
        return 0

    if '-dumpversion' in arguments:
        print(VERSION)
        return 0

    for arg in arguments:
        canonical = arg.split('=', maxsplit=1)[0]
        if canonical.startswith('-'):
            recognized = canonical in recognized_arguments or \
                any(canonical.startswith(x) for x in recognized_beginnings)
            if not recognized:
                print('Error: unrecognized argument "{0}"'.format(arg),
                      file=sys.stderr)
                return 1

    if '-o' in arguments or '--output' in arguments:
        idx = arguments.index('-o' if '-o' in arguments else '--output')
        outfile = arguments[idx + 1]
        open(outfile, 'a').close()  # create an empty file if it does not exist

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
