#!/usr/bin/python3

from subprocess import call, check_output
from os import environ
import sys
from datetime import datetime



hostinfo = [['u0422778@kingspeak.chpc.utah.edu', 12],
            ['sawaya@bihexal.cs.utah.edu', 24],
            ['sawaya@gaussr.cs.utah.edu', 4]]

#constants
git = '/usr/bin/git'
psql = check_output('which psql', shell=True)[:-1]
dumpFile = 'db/db_backup/dumpfile'

if len(sys.argv) > 1:
    hostinfo.append([sys.argv[1], int(sys.argv[2])])
    
## first: check if git is current (git diff-index --quiet HEAD != 0)    
## if not current, exit and require commit / push
if call([git, 'diff-index', '--quiet', 'HEAD', dumpFile]) != 0:
    print('Please commit git with ' + dumpFile + ' before continuing')
    sys.exit(1)
f = open(dumpFile, 'w')
## do db backup
if call(['/usr/lib/postgresql/9.4/bin/pg_dump', 'qfp'], stdout=f) != 0:
    print('db backup failed.  please correct problem before continuing . . .')
    sys.exit(1)

## do git add dumpfile && git commit -m 'db auto-backup'
if (call([git, 'add', 'db/db_backup/dumpfile']) != 0 or
    call([git, 'commit', '-m "db auto-backup"']) != 0):
    print('problem committing new db backup file')

## rename tests (tests_[time])
## create table tests like [backup table name]
newTable = 'tests_' + str(datetime.now().microsecond)
if (call([psql, '-d', 'qfp', '-c', 'ALTER TABLE tests RENAME TO ' + newTable + ';']) != 0 or
    call([psql, '-d', 'qfp', '-c', 'CREATE TABLE tests (like ' + newTable + ');']) != 0):
    print('failure renaming old table or creating empty clone')
    sys.exit(1)

## import
for h in hostinfo:
    print('collecting data from ' + h[0])
    call(['ssh', h[0], 'if [[ ! -e remote_qfp ]]; then ' +
          'mkdir remote_qfp && cd remote_qfp && ' +
          'git clone https://github.com/geof23/qfp && ' +
          'cd .. ; ' +
          'fi && ' +
          'cd remote_qfp/qfp && ' +
          'git stash && ' +
          'git checkout master ' +
          'git pull ' +
          './hostCollect.sh ' + str(h[1])])
