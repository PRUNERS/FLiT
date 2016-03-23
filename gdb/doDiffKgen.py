#!/usr/bin/env python3

# this is the launcher for QFP_gdb
# This is to run KGen kernels with QFPD

import sys
from subprocess import check_output
import os

def usage():          #  1      2       3                   4
    print(sys.argv[0] + '[exe0] [exe1] [qc_file | divFile] [emacsNoWindow = t|f]')


if len(sys.argv) < 4 or len(sys.argv) > 5:
    usage()
    exit(1)

cmd_out = ""
emacs = check_output('which emacs', shell=True)[:-1]  #remove newline from which
ln = check_output('which ln', shell=True)[:-1]

topDir = os.path.dirname(__file__)
inf1 = os.path.realpath(sys.argv[1])
inf2 = os.path.realpath(sys.argv[2])
os.environ['PARAMS'] = sys.argv[3]

NW = True
if len(sys.argv) == 5:
    if sys.argv[7] == 'f':
        NW = False

sys.path.append(topDir)

os.chdir(topDir) #this has secret .gdbinit script


try:
    print('creating symlinks to binaries...')
    print(check_output([ln, '-sf', inf1, 'inf1']))
    print(check_output([ln, '-sf', inf2, 'inf2']))
    print('--eval="(gdb \"gdb -i=mi\")"')
    cmd = [emacs, '--eval=(gdb "gdb -i=mi")']
    if NW:
        cmd.append('-nw')
    cmd_out = check_output(cmd)
    
except CalledProcessError:
    print(cmd_out)
    exit(1)

exit(0)
