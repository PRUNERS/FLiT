#!/usr/bin/python3

from subprocess import call, check_output
from os import chdir, getcwd, remove, environ
import sys
import datetime
import glob


hostinfo = [
#            ['u0422778@kingspeak2.chpc.utah.edu', 12],
#            ['sawaya@bihexal.cs.utah.edu', 24],
#            ['sawaya@gaussr.cs.utah.edu', 8],
            ['sawaya@ms0141.utah.cloudlab.us', 8]
]
#['u0422778@kingspeak2.chpc.utah.edu', 12],

#constants
git = check_output('which git', shell=True)[:-1]
psql = check_output('which psql', shell=True)[:-1]
notes = ''
verbose = ''
if environ.get("VERBOSE") == 'verbose':
    verbose = 'verbose'
    

def usage():
    print('usage: ' + sys.argv[0] + ' notes [optional: fqdn procs, ... ]')
    print('where fqdn = host to run tests on; procs is number of processes to use')

if len(sys.argv) > 1:
    notes = sys.argv[1]
    for x in range(2, len(sys.argv), 2):
        hostinfo.append([sys.argv[x], int(sys.argv[x+1])])
else:
    usage()
    exit(1)

# for f in glob.iglob('results/*'):
#     remove(f);


for h in hostinfo:
    print('collecting data from ' + h[0])
    stdo = check_output(['ssh', h[0], 'if [[ -e remote_qfp ]]; then ' +
                         'rm -fr remote_qfp; ' +
                         'fi && ' +
                         'mkdir remote_qfp && cd remote_qfp && ' +
                         'git clone https://github.com/geof23/qfp && ' +
                         'cd qfp && ' +
                         'git checkout rel_lt && ' +
                         'git pull && ' +
                         'VERBOSE=' + verbose + ' ./hostCollect.sh ' + str(h[1])])
    print(stdo)
    if verbose == '':
        stdo = check_output(['scp', h[0] + ':~/remote_qfp/qfp/results/masterRes*', 'results/masterRes' + h[0]])
        print(stdo)

if verbose == '':
    chdir('results')
    stdo = check_output([psql, '-d', 'qfp', '-c', 'INSERT INTO runs (rdate, notes) VALUES (\'' + str(datetime.date.today())
                         + '\', \'' + notes + '\');'])
    for f in glob.iglob('masterRes*'):
        stdo = check_output([psql, '-d', 'qfp', '-c', 'select importQFPResults(\'' + getcwd() + '/' + f + '\');'])
        print(stdo)
    
