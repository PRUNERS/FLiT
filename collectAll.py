#!/usr/bin/env python3

from subprocess import call, check_output
from os import chdir, getcwd, remove, environ, path
import sys
import datetime
import glob

import prepDBHost

home_dir = path.dirname(path.realpath(__file__))

#constants
git = check_output('which git', shell=True)[:-1]
psql = check_output('which psql', shell=True)[:-1]

#vars
notes = ''
DBINIT = 'prepDBHost.py'
BRANCH = 'unified_script'

def usage():
    print('usage: ' + sys.argv[0] + ' "notes"')
    print('\tyou must populate ' + home_dir + '/hostfile.py with')
    print('\trun and db host info (see file for details)')


if len(sys.argv) == 2:
    notes = sys.argv[1]
    try:
        import hostfile
    except ImportError:
        print('missing hostfile.py')
        usage()
        exit(1)
else:
    usage()
    exit(1)

run_hosts = hostfile.RUN_HOSTS
db_host = hostfile.DB_HOST

#clear space on db host -- copy and exec prepDBHost.py
print('preparing workspace on DB server, ' + db_host[1] + '...')
print(check_output(['scp', DBINIT, db_host[0] + '@' + db_host[1] + ':~/']))
print(check_output(['ssh', db_host[0] + '@' + db_host[1],
                    './' + DBINIT]))

#Now it's time to do the 
for h in run_hosts:
    print('collecting data from ' + h[1])
    stdo = check_output([
        'ssh', h[0] + '@' + h[1],
        'if [[ -e /tmp ]]; then cd /tmp; fi && ' +
        'if [[ -e remote_qfp ]]; then ' +
        'rm -fr remote_qfp; ' +
        'fi && ' +
        'mkdir remote_qfp && cd remote_qfp && ' +
        'git clone https://github.com/geof23/qfp && ' +
        'cd qfp && ' +
        'git checkout ' + BRANCH + ' && ' +
        'git pull && ' +
        './hostCollect.sh ' +
        str(h[2]) + ' ' + str(h[3])
    ])
    print(stdo)
    #copy results to DB server
    stdo = check_output([
        'scp',
        h[0] + '@' + h[1] + ':~/remote_qfp/qfp/results/*.tgz',
        db_host[0] + '@' + db_host[1] + ':~/' + prepDBHost.COLL_DIR
    ])
    print(stdo)

#unpack result files @ db server
stdo = check_output(['ssh', db_host[0] + '@' + db_host[1],
                     'cd ' + COLL_DIR + ' && ' +
                     'for f in *.tgz; do tar xf $f; done'])
print(stdo)
#now import results into database
stdo = check_output(['ssh', db_host[0] + '@' + db_host[1],
                     "psql qfp -c dofullflitimport('" +
                     prepDBHost.COLL_DIR + "','" +
                     notes + "')"
                     ])
print(stdo)
#done!

print('data collection is complete, results stored in DB @ ' + db_host[1])
