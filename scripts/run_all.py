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
        new_env = os.environ.copy()
        new_env['SSHPASS'] = host[1]
        new_env['CUDA_ONLY'] = host[0][4]
        new_env['DO_PIN'] = host[0][5]
        new_env['CORES'] = host[0][2]
        new_env['DB_HOST'] = db_host[1]
        new_env['DB_USER'] = db_host[0]
        procs.append(Popen('sshpass -e ' + 
                           host[2].format(host[0][0], host[0][1]),
                           shell=True, stdout=PIPE, env=new_env))
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
new_env = os.environ.copy()
new_env['SSHPASS'] = db_pw
print(check_output(['sshpass', '-e', 'scp', home_dir + '/' + DBINIT,
                    db_host[0] + '@' + db_host[1] + ':~/', '&&',
                    'ssh', db_host[0] + '@' + db_host[1], ' ./' + DBINIT], 
                   env=new_env))

#get run# from db
print(check_output(['sshpass', '-e', 'ssh', 
              db_host[0] + '@' + db_host[1] +
              ' psql flit -t -c "insert into runs (rdate, notes) ' +
              'values (\'$(date)\', \'' + notes + '\')"'], env=new_env))
run_num = int(check_output(['sshpass', '-e', 'ssh', 
              db_host[0] + '@' + db_host[1] +
              ' psql flit -t -c "select max(index) from runs"'], env=new_env))

#launch workers
pwds = []
print('please enter your credentials (pwds) for RUN_HOSTS, ' +
      '[or empty for passphraseless ssh key auth]. No creds will be stored')
for host in run_hosts:
    pwds.append(getpass.getpass('Enter pwd for ' + host[1] + ':'))

cmds = []
os.environ['REPO'] = 'https://github.com/geof23/qfp'
os.environ['BRANCH'] = BRANCH
os.environ['FLIT_DIR'] = 'QFP'
for host in run_hosts:
    if hosts[3] is None:
        cmd = ('ssh {0}@{1} "export TMPD=$(mktemp -d) && ' +
               'cd ${TMPD} && ' +
               'git clone -b ${BRANCH} --recursive ${REPO} && '
               'cd ${FLIT_DIR} && scripts/hostCollect.sh"')
    else:
        cmd += (
            'scp ' + home_dir + '/{2} {0}@{1}:/tmp && ' +
            'ssh {0}@{1} "sbatch /tmp/' + hosts[3] +'"'
            )
    cmds.append(cmd)
runOnAll(cmds, pwds)

#import to database -- need to unzip and then run importqfpresults2
new_env = os.environ.copy()
new_env['SSHPASS'] = db_pw

cmd = (
    'cd ~/flit_data && ' +
    'for f in *.tgz; do tar xf $f; done && ' +
    'psql flit -c "select importqfpresults2(\'$(pwd)\',' + str(run_num) + ')" '
    )
if any(zip*(run_hosts)[5]): #any opcode collections
    cmd += (
        '&& rm *O0 *O1 *O2 *O3 *out_ ' + 
        '&& psql flit -c "select importopcoderesults(\'$(pwd)\',' + str(run_num) +
        ')" && echo $? && echo "db importation complete"'
        )
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1] +
                    ' ' + cmd ], env=new_env))

#display report / exit message
cmd = (
    'mkdir -p ~/flit_data/reports && ' +
    'cd ~/flit_data/reports && ' +
    'touch f_all.pdf d_all.pdf e_all.pdf && ' +
    'psql flit -c "select createschmoo(' + str(run_num) + ',' +
    '\'{\"f\"}\',\'{\"icpc\", \"g++\", \"clang++\"}\',' + 
    '\'{\"O1\", \"O2\", \"O3\"}\',' +
    '\'\',3,\'$(pwd)/f_all.pdf\')" && ' +
    'psql flit -c "select createschmoo(' + str(run_num) + ',' +
    '\'{\"d\"}\',\'{\"icpc\", \"g++\", \"clang++\"}\',' + 
    '\'{\"O1\", \"O2\", \"O3\"}\',' +
    '\'\',3,\'$(pwd)/d_all.pdf\')" && ' +
    'psql flit -c "select createschmoo(' + str(run_num) + ',' +
    '\'{\"e\"}\',\'{\"icpc\", \"g++\", \"clang++\"}\',' + 
    '\'{\"O1\", \"O2\", \"O3\"}\',' +
    '\'\',3,\'$(pwd)/e_all.pdf\')"'
)
print(check_output(['sshpass', '-e', 'ssh', db_host[0] + '@' + db_host[1] +
                    ' ' + cmd], env=new_env))
