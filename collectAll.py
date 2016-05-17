#!/usr/bin/python3

from subprocess import call, check_output
from os import chdir, getcwd, remove, environ
import sys
import datetime
import glob


hostinfo = [['u0422778@kingspeak.chpc.utah.edu', 12],
            ['sawaya@bihexal.cs.utah.edu', 24],
            ['sawaya@gaussr.cs.utah.edu', 4],
            ['sawaya@ms0123.utah.cloudlab.us', 1]]
            # [environ["USER"] + 'sawaya@bihexal.cs.utah.edu', 24],
            # [environ["USER"] + 'sawaya@gaussr.cs.utah.edu', 4]]

#constants
git = '/usr/bin/git'
psql = check_output('which psql', shell=True)[:-1]
dumpFile = 'db/db_backup/dumpfile'
notes = ''
#sed = '/bin/sed'
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

# kp_name = input('Enter your username for kingspeak: ')
# hostinfo[0][0] = kp_name + hostinfo[0][0]


for f in glob.iglob('results/*'):
    remove(f);

## first: check if git is current (git diff-index --quiet HEAD != 0)    
## if not current, exit and require commit / push
# if call([git, 'diff-index', '--quiet', 'HEAD', dumpFile]) != 0:
#     print('Please commit git with ' + dumpFile + ' before continuing')
#     sys.exit(1)
# f = open(dumpFile, 'w')
## do db backup
# if call(['/usr/lib/postgresql/9.4/bin/pg_dump', 'qfp'], stdout=f) != 0:
#     print('db backup failed.  please correct problem before continuing . . .')
#     sys.exit(1)

## do git add dumpfile && git commit -m 'db auto-backup'
# if (call([git, 'add', 'db/db_backup/dumpfile']) != 0 or
#     call([git, 'commit', '-m "db auto-backup"']) != 0):
#     print('problem committing new db backup file')

## rename tests (tests_[time])
## create table tests like [backup table name]
# newTable = 'tests_' + str(datetime.now().microsecond)
# if (call([psql, '-d', 'qfp', '-c', 'CREATE TABLE ' + newTable + ' AS TABLE tests;']) != 0 or
#     call([psql, '-d', 'qfp', '-c', 'DELETE FROM tests);']) != 0):
#     print('failure creating backup table for tests or deleting * from tests')
#     sys.exit(1)

## import
for h in hostinfo:
    print('collecting data from ' + h[0])
    stdo = check_output(['ssh', h[0], 'if [[ ! -e remote_qfp ]]; then ' +
          'mkdir remote_qfp && cd remote_qfp && ' +
          'git clone https://github.com/geof23/qfp && ' +
          'cd .. ; ' +
          'fi && ' +
          'cd remote_qfp/qfp && ' +
          'git stash && ' +
#          'git checkout master  && ' +
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
    
