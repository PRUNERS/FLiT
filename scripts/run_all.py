#!/usr/bin/env python3

# this is the master run test
# by executing this file, after having configured
# hostfile.py (and any corresponding collect scripts),
# will
## configure DB server
## push tests out to workers
## run tests on workers -- pushing data to db server
## collect data into db

import os
from subprocess import Popen, PIPE, check_output, STDOUT
import getpass
import sys
from datetime import datetime

#local
import hostfile

home_dir = os.path.dirname(os.path.realpath(__file__))

#vars
notes = ''
DBINIT = 'prepDBHost.py'
BRANCH = 'unified_script'
db_host = hostfile.DB_HOST
run_hosts = hostfile.RUN_HOSTS
REPO = 'https://github.com/geof23/qfp'
FLIT_DIR = 'qfp'

def usage():
    print('usage: ' + sys.argv[0] + ' "notes"')
    print('\tyou must populate ' + home_dir + '/hostfile.py with')
    print('\trun and db host info (see file for details)')

def makeEnvStr(env):
    retVal = ''
    for k,v in env.items():
        retVal += 'export ' + k + '=' + v + '; '
    return retVal
    
def runOnAll(cmdStrs, pwds):
    procs = []
    for host in zip(run_hosts, pwds, cmdStrs):
        local_env = os.environ.copy()
        local_env['SSHPASS'] = str(host[1])
        rem_env = {}
        rem_env['CUDA_ONLY'] = str(host[0][4])
        rem_env['DO_PIN'] = str(host[0][5])
        rem_env['CORES'] = str(host[0][2])
        rem_env['DB_HOST'] = str(db_host[1])
        rem_env['DB_USER'] = str(db_host[0])
        cmdStr = host[2].format(host[0][0], host[0][1], host[0][3], makeEnvStr(rem_env))
        print('executing: ' + cmdStr)
        procs.append(Popen('sshpass -e ' + 
                           cmdStr,
                           shell=True, stdout=PIPE, stderr=STDOUT,
                           env=local_env))
    for p in procs:
        p.wait()
        print(p.stdout.read())
        
#check hostfile.py

print('we\'re here')

if len(sys.argv) == 2:
    notes = sys.argv[1]

else:
    usage()
    exit(1)


#setup db
print('preparing workspace on DB server, ' + db_host[1] + '...')
db_pw = getpass.getpass('Enter password for ' + db_host[1]+ ': ')
new_env = os.environ.copy()
new_env['SSHPASS'] = db_pw
print(check_output(['sshpass', '-e', 'scp', home_dir + '/' + DBINIT,
                    db_host[0] + '@' + db_host[1] + ':~/'], env=new_env))
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1],
                    ' ./' + DBINIT], env=new_env))

#get run# from db
print(check_output(['sshpass', '-e', 'ssh', 
                    db_host[0] + '@' + db_host[1],
              'psql flit -t -c "insert into runs (rdate, notes) ' +
              'values (\'' + str(datetime.now()) + '\', \'' + notes + '\')"'],
                   env=new_env))
run_num = int(check_output(['sshpass', '-e', 'ssh', 
                            db_host[0] + '@' + db_host[1],
                            'psql flit -t -c "select max(index) from runs"'],
                           env=new_env))

#launch workers
pwds = []
print('please enter your credentials (pwds) for RUN_HOSTS, ' +
      '[or empty for passphraseless ssh key auth]. No creds will be stored')
for host in run_hosts:
    pwds.append(getpass.getpass('Enter pwd for ' + host[1] + ': '))


cmds = []
#send run host scripts, if used
for host in run_hosts:
    if host[3] is not None: #0-user 1-host 2-script 3-enviro
        print('in copy loop, host[3] is: ' + str(host[3]))
        cmd = ('scp ' + home_dir + '/{2} {0}@{1}:')
    else:
        cmd = ('ssh {0}@{1} exit') #dummy -- this is a hack
    cmds.append(cmd)
runOnAll(cmds, pwds)

cmds = []
for host in run_hosts:
    if host[3] is None: #0-user 1-host 2-script 3-enviro
        cmd = ('ssh {0}@{1} "{3} export TMPD=\$(mktemp -d) && ' +
                'cd \${{TMPD}} && ' +
                'git clone -b ' + BRANCH + ' --recursive ' + REPO + ' && '
                'cd ' + FLIT_DIR + ' && scripts/hostCollect.sh"')
    else:
        cmd = ('ssh {0}@{1} "{3} sbatch ' + host[3] +'"')
    cmds.append(cmd)
runOnAll(cmds, pwds)

#import to database -- need to unzip and then run importqfpresults2
new_env = os.environ.copy()
new_env['SSHPASS'] = db_pw

cmd = (
    'cd ~/flit_data && ' +
    'for f in *.tgz; do tar xf $f; done && ' +
    'psql flit -c "select importqfpresults2(\'\$(pwd)\',' + str(run_num) + ')" '
    )
if any(list(zip(*run_hosts))[5]): #any opcode collections
    cmd += (
        '&& rm *O0 *O1 *O2 *O3 *out_ ' + 
        '&& psql flit -c "select importopcoderesults(\'\$(pwd)\',' + str(run_num) +
        ')" && echo \$? && echo "db importation complete"'
        )
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1],
                    '"' + cmd + '"' ], env=new_env))

#display report / exit message
cmd = (
    'mkdir -p ~/flit_data/reports && ' +
    'cd ~/flit_data/reports && ' +
    'touch f_all.pdf d_all.pdf e_all.pdf && ' +
    'psql flit -c \"select createschmoo(' + str(run_num) + ',' +
    '\'{\\"f\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
    '\'{\\"O1\\", \\"O2\\", \\"O3\\"}\',' +
    '\'\',3,\'$(pwd)/f_all.pdf\')\" && ' +
    'psql flit -c \"select createschmoo(' + str(run_num) + ',' +
    '\'{\\"d\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
    '\'{\\"O1\\", \\"O2\\", \\"O3\\"}\',' +
    '\'\',3,\'$(pwd)/d_all.pdf\')\" && ' +
    'psql flit -c \"select createschmoo(' + str(run_num) + ',' +
    '\'{\\"e\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
    '\'{\\"O1\\", \\"O2\\", \\"O3\\"}\',' +
    '\'\',3,\'\$(pwd)/e_all.pdf\')\"'
)
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1],
                    '"' + cmd + '"'], env=new_env))
