#!/usr/bin/env python3

# this is the launcher for QFP_gdb

import sys
from subprocess import check_output
import os

def usage():             #  1         2           3        4              5                   6
    print(sys.argv[0] + '[bin1 path] [bin2 path] [testid1] [prec1 = f|d|e] [sort1 = lt|gt|ns|bi] [testid2] [prec2] [sort2] [emacsNoWindow = t|f]')


if len(sys.argv) < 8 or len(sys.argv) > 9:
    usage()
    exit(1)

cmd_out = ""
emacs = check_output('which emacs', shell=True)[:-1]  #remove newline from which
ln = check_output('which ln', shell=True)[:-1]

topDir = os.path.dirname(__file__)
inf1 = os.path.realpath('perpVects/' + sys.argv[1])
inf2 = os.path.realpath('perpVects/' + sys.argv[2])
os.environ['TEST1'] = sys.argv[3]
os.environ['PRECISION1'] = sys.argv[4]
os.environ['SORT1'] = sys.argv[5]
os.environ['TEST2'] = sys.argv[6]
os.environ['PRECISION2'] = sys.argv[7]
os.environ['SORT2'] = sys.argv[8]
os.environ['NO_WATCH'] = 'false'
NW = True
if len(sys.argv) >= 10:
    if sys.argv[9] == 'f':
        NW = False

sys.path.append(topDir)

os.chdir(topDir) #this has secret .gdbinit script


try:
    print('creating symlinks to binaries...')
    print(check_output([ln, '-sf', inf1, 'inf1']))
    print(check_output([ln, '-sf', inf2, 'inf2']))
    print('--eval="(gdb \"gdb -i=mi\")"')
    cmd = [emacs, '--eval=(gdb "gdb -i=mi inf1")']
    if NW:
        cmd.append('-nw')
    cmd_out = check_output(cmd)
    
except CalledProcessError:
    print(cmd_out)
    exit(1)

exit(0)
