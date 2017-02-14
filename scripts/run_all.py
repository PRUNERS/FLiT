# this is the master run test
# by executing this file, after having configured
# hostfile.py (and any corresponding collect scripts),
# will
## configure DB server
## push tests out to workers
## run tests on workers -- pushing data to db server
## collect data into db

import os
from subprocess import Popen, PIPE
import getpass

import prepDBHost

home_dir = path.dirname(path.realpath(__file__))

#constants
# git = check_output('which git', shell=True)[:-1]
# psql = check_output('which psql', shell=True)[:-1]

#vars
notes = ''
DBINIT = 'prepDBHost.py'
BRANCH = 'unified_script'
db_host = hostfile.DB_HOST

def usage():
    print('usage: ' + sys.argv[0] + ' "notes"')
    print('\tyou must populate ' + home_dir + '/hostfile.py with')
    print('\trun and db host info (see file for details)')

def runOnAll(cmdStrs, pwds):
    procs = []
    for host in zip(run_hosts, pwds, cmdStrs):
        os.environ['SSHPASS'] = host[1]
        os.environ['CUDA_ONLY'] = host[0][4]
        os.environ['DO_PIN'] = host[0][5]
        os.environ['CORES'] = host[0][2]
        os.environ['DB_HOST'] = db_host[1]
        os.environ['DB_USER'] = db_host[0]
        procs.append(Popen('sshpass -e ssh ' + host[0][0] +
                           '@' + host[0][1] + ' ' + host[2],
                           shell=True, stdout = PIPE))
    for p in procs:
        p.wait()
        print(p.stdout.read())
        
#check hostfile.py

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

#setup db
print('preparing workspace on DB server, ' + db_host[1] + '...')
db_pw = getpass.getpass('Enter password for ' + db_host[1])
os.environ['SSHPASS'] = db_pw
print(check_output(['sshpass', '-e', 'scp', home_dir + '/' + DBINIT,
                    db_host[0] + '@' + db_host[1] + ':~/']))
print(check_output(['sshpass', '-e', 'ssh',
                    db_host[0] + '@' + db_host[1], ' ./' + DBINIT]))

#get run# from db
print(check_output(['sshpass', '-e', 'ssh', 
              db_host[0] + '@' + db_host[1] +
              ' psql qfp -c "insert into runs (rdate, notes) ' +
              'values (\'$(date)\', \'' + notes + '\')"']))
run_num = int(check_output(['sshpass', '-e', 'ssh', 
              db_host[0] + '@' + db_host[1] +
              ' psql qfp -c "select max(index) from runs"']))

#launch workers
pwds = []
print('please enter your credentials (pwds) for RUN_HOSTS, ' +
      '[or empty for passphraseless ssh key auth]. No creds will be stored')
for host in run_hosts:
    pwds.append(getpass.getpass('Enter pwd for ' + host[1] + ':'))

#setup run hosts with repo
runOnAll(['if [[ -e /tmp ]]; then cd /tmp; fi && ' +
         'if [[ -e remote_qfp ]]; then ' +
         'rm -fr remote_qfp; ' +
         'fi && ' +
         'mkdir remote_qfp && cd remote_qfp && ' +
         'git clone https://github.com/geof23/qfp && ' +
         'cd qfp && ' +
         'git checkout ' + BRANCH + ' && ' +
         'git pull'] * len(run_hosts), pwds)

cmds = []
for host in run_hosts:
    if hosts[3] is None:
        cmds.append('cd remote_qfp && scripts/hostCollect.py')
    else:
        cmds.append('sbatch scripts/' + hosts[3])
runOnAll(cmd, pwds)

#import to database -- need to unzip and then run importqfpresults2

cmd = (
    'cd ~/flit_data && ' +
    'for f in *.tgz; do tar xf $f; done && ' +
    'psql flit -c "select importqfpresults2(\'$(pwd)\',' + str(run_num) + ')"' +
    ' && psql flit -c "select importopcoderesults(\'$(pwd)\',' + str(run_num) +
    ')" && echo $? && echo "db importation complete"'
)

os.environ['SSHPASS'] = db_pw
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1] +
                    ' ' + cmd ]))

#display report / exit message
cmd = (
    'mkdir -p ~/flit_data/reports && ' +
    'cd ~/flit_data/reports && ' +
    'touch f_all.pdf d_all.pdf e_all.pdf && ' +
    'psql flit -c "select createschmoo(' + str(run_num) + 
)

print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1] +
                    ' ' + 
