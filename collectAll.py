#!/usr/bin/python3

from subprocess import call
from os import environ
import sys



hostinfo = [['u0422778@kingspeak.chpc.utah.edu', 12],
            ['sawaya@bihexal.cs.utah.edu', 24]]

if(len(sys.argv) > 1):
    hostinfo.append([sys.argv[1], 8])

for h in hostinfo:
    print('collecting data from ' + h[0])
    call('ssh', h[0], 'if [[ ! -e qfp ]]; then git clone https://github.com/' +
         'geof23/qfp && rm "qfp/results/*" ; cd qfp && git checkout dumpSQL ' +
         '&& git pull && cd ' +
         'perpVects && make -j ' + str(h[1]) + ' -f Makefile2 && ../' +
         'hostCollect.sh')
    )
