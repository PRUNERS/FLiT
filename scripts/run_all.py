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
from subprocess import Popen, PIPE, check_output, STDOUT, CalledProcessError
import getpass
import sys
from datetime import datetime
import re
import time

#local
import hostfile

home_dir = os.path.dirname(os.path.realpath(__file__))

#vars
notes = ''
DB_HOST_AUX = '/tmp/flitDbDir'
DBINIT = 'prepDBHost.py'
db_host = hostfile.DB_HOST
run_hosts = hostfile.RUN_HOSTS
FLIT_DIR = 'qfp'
REM_ENV = {'FLIT_DIR': FLIT_DIR}
SSHS = 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -q '
SSHL = ['ssh', '-o UserKnownHostsFile=/dev/null', '-o StrictHostKeyChecking=no', '-q']
SCPS = 'scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -q '
SCPL = ['scp', '-o UserKnownHostsFile=/dev/null', '-o StrictHostKeyChecking=no', '-q']
pwds = {}

def usage():
    print('usage: ' + sys.argv[0] + ' "notes"')
    print('\tyou must populate ' + home_dir + '/hostfile.py with')
    print('\trun and db host info (see file for details)')

def makeEnvStr(env):
    retVal = ''
    for k,v in env.items():
        retVal += 'export ' + k + '=' + v + '; '
    return retVal

def getJobNum(substr):
    m = re.match(r".*Submitted batch job ([0-9]+)",
                 substr, re.DOTALL)
    try:
        return m.group(1)
    except AttributeError:
        return None

def slurmWait(user, host, pw, jobn):
    envir = os.environ.copy()
    envir['SSHPASS'] = pw
    try:
        while user in check_output(['sshpass', '-e', *SSHL, 
                                    user + '@' + host,
                                    'squeue -j ' + jobn],
                                   env=envir).decode("utf-8"):
            print('.')
            time.sleep(10)
    except CalledProcessError:
        pass

def runOnAll(cmdStrs):
    procs = []
    results = []
    for host in zip(run_hosts, cmdStrs):
        pkey = host[0][0]+'@'+host[0][1]
        local_env = os.environ.copy()
        local_env['SSHPASS'] = pwds[pkey]
        rem_env = REM_ENV.copy()
        rem_env['CUDA_ONLY'] = str(host[0][4])
        rem_env['DO_PIN'] = str(host[0][5])
        rem_env['CORES'] = str(host[0][2])
        rem_env['DB_HOST'] = str(db_host[1])
        rem_env['DB_USER'] = str(db_host[0])
        rem_env['SLURMED'] = str(host[0][3])
        if host[1] is not None:
            cmdStr = ('sshpass -e ' +
                      host[1].format(host[0][0], host[0][1],
                                     host[0][3],
                                     makeEnvStr(rem_env)))
            print('executing: ' + cmdStr)
            procs.append([Popen(cmdStr,
                                shell=True, stdout=PIPE, stderr=STDOUT,
                                env=local_env),
                          host[0][3], host[0][0], host[0][1], pwds[pkey]])
    for p in procs:
        p[0].wait()
        outs = p[0].stdout.read().decode("utf-8")
        results.append(outs.strip())
        print(outs)
        if p[1] is not None: #if it's slurm
            jn = getJobNum(outs)
            if jn is not None:
                slurmWait(p[2], p[3], p[4], jn)
    return results

def getPasswords():
    print('please enter your credentials (pwds) for remote hosts, ' +
          '[or empty for passphraseless ssh key auth]. No creds will be stored')
    print(db_host)
    print(run_hosts)
    for host in ((db_host,) + run_hosts):
        key = host[0] + '@' + host[1]
        if key not in pwds:
            pwds[key] = getpass.getpass(
                'Enter password for ' + host[0] + '@' + host[1] + ': '
            )

if len(sys.argv) == 2:
    notes = sys.argv[1]

else:
    usage()
    exit(1)


#get passwords
getPasswords()

#setup db -- we're doing this first because it's cheap and if it fails,
#the rest of the work will go to waste
print('preparing workspace on DB server, ' + db_host[1] + '...')
os.chdir(home_dir + '/../db')
print(check_output('tar zcf ' + home_dir + '/dbPy.tgz *',
                   shell=True).
      decode("utf-8"))
os.chdir(home_dir) 
new_env = os.environ.copy()
new_env['SSHPASS'] = pwds[db_host[0] + '@' + db_host[1]]
print(check_output('sshpass ' + SCPS + home_dir + '/dbPy.tgz ' +
                   db_host[0] + '@' + db_host[1] + ':', shell=True,
                   env=new_env).
      decode("utf-8"))
os.remove('dbPy.tgz')
print(check_output(['sshpass', '-e', *SCPL, home_dir + '/' + DBINIT,
                    db_host[0] + '@' + db_host[1] + ':~/'],
                   env=new_env).decode("utf-8"))
print(check_output(['sshpass', '-e', *SSHL, db_host[0] + '@' + db_host[1],
                    ' ./' + DBINIT, DB_HOST_AUX], env=new_env).decode("utf-8"))

# #get run# from db
print(check_output(['sshpass', '-e', *SSHL, 
                    db_host[0] + '@' + db_host[1],
              'psql flit -t -c "insert into runs (rdate, notes) ' +
              'values (\'' + str(datetime.now()) + '\', \'' + notes + '\')"'],
                   env=new_env).decode("utf-8"))
run_num = int(check_output(['sshpass', '-e', *SSHL, 
                            db_host[0] + '@' + db_host[1],
                            'psql flit -t -c "select max(index) from runs"'],
                           env=new_env).decode("utf-8"))

#create worker package (use to be git checkout)
print('creating worker package . . .')
olddir = os.getcwd()
os.chdir(home_dir + '/..')
print(check_output(['tar', 'zcf', 'flit.tgz', 'scripts', 'src', 'Makefile.switches', 'Makefile.switches.cuda']).decode("utf-8"))
os.chdir(olddir)
print('done.')



cmds = []
#transfer package to workers (or portal, if slurm script provided)
package_dirs = runOnAll([SSHS + ' {0}@{1} "echo \$(mktemp -d -p ~/)"'] *
                        len(run_hosts))
print(*package_dirs, sep=',')
runOnAll([SCPS + home_dir + '/../flit.tgz {0}@{1}:' + d for d in package_dirs])
for i, d in enumerate(package_dirs):
    assert len(d) > 0, "got zero length directory from mktemp %r" % run_hosts[i][1]

#this does all the work of running and collecting

cmds = []
copycs = []
cleancs = []
for host in zip(run_hosts, package_dirs):
    if host[0][3] is None: #0-user 1-host 2-script 3-enviro
        cmd = (SSHS + ' {0}@{1} "{3} cd ' + host[1] + ' && ' +
               'tar xf flit.tgz && scripts/hostCollect.sh"'
               )
        copyc = (SCPS + '{0}@{1}:/' + host[1] + '/results/"*.tgz" .')
        cleanc = None #DELME
        #cleanc = ('ssh {0}@{1} "rm -fr ' + host[1] + '"')
    else:
        cmd = (SSHS + ' {0}@{1} "cd ' + host[1] + ' && ' +
               'tar xf flit.tgz scripts/' + host[0][3] + ' && ' +
               'cd .. && ' +
               'export PDIR=' + host[1] + ' {3} sbatch ' +
               host[1] + '/scripts/' + host[0][3] + '"')
        copyc = None
        cleanc = None
    cmds.append(cmd)
    copycs.append(copyc)
    cleancs.append(cleanc)
runOnAll(cmds)
runOnAll(copycs)
runOnAll(cleancs)
 
new_env = os.environ.copy()
new_env['SSHPASS'] = pwds[db_host[0] + '@' + db_host[1]]

# #copy the result files that we copied to here (the launch host)
print(check_output('sshpass -e ' + SCPS + home_dir +
                    '/*.tgz ' + db_host[0] + '@' +
                    db_host[1] + ':flit_data',
                   shell=True,env=new_env).decode("utf-8"))

print(check_output('rm *.tgz', shell=True).decode("utf-8"))

#import to database -- need to unzip and then run importqfpresults2
cmd = (
    'cd ~/flit_data && ' +
    'for f in *.tgz; do tar xf \$f; done && ' +
    'psql flit -c \\"select importqfpresults2(\'\$(pwd)\',' +
    str(run_num) + ')\\" '
    )
if any(list(zip(*run_hosts))[5]): #any opcode collections
    cmd += (
        '&& rm *O0 *O1 *O2 *O3 *out_ ' + 
        '&& psql flit -c \\"select importopcoderesults(\'\$(pwd)\',' +
        str(run_num) +
        ')\\" && echo \$? && echo \\"db importation complete\\"'
        )
print(check_output('sshpass -e ' + SSHS + db_host[0] + '@' + db_host[1] +
                    ' "' + cmd + '"', env=new_env,
                  shell=True).decode("utf-8"))

#display report / exit message


run_num = 1

plot_dir = (check_output(['sshpass', '-e', *SSHL, db_host[0] + '@' + db_host[1],
                          'echo $HOME'], env=new_env).decode("utf-8").strip() +
            '/flit_data/reports')

#get list of hosts -- are often different than the interactive node used
#to submit the work


cmd = ('psql flit -t -c \\"select distinct host from tests where run = ' +
       str(run_num) + '\\"')
rhosts = [s.strip() for s in check_output('sshpass -e ' + SSHS + db_host[0] +
                                          '@' + db_host[1] +
                                          ' "' + cmd + '"',
                                          env=new_env,
                                          shell=True).
          decode("utf-8").split('\n') if len(s) > 0] 
# print('rhosts is: ')
# print(*rhosts, sep=',')

pcmd = (
    'set -x && ' +
    'mkdir -p ~/flit_data/reports && ' +
    'cd ~/flit_data/reports && ' +
    'chmod 777 ~/flit_data/reports '
)

gcmd = ('set -x && '
        # 'unset MATPLOTLIBRC && '
        # 'export MATPLOTLIBRC=' + DB_HOST_AUX + ' && ' +
        # 'echo \$MATPLOTLIBRC && '
)

for h in rhosts:
    pcmd += (
        ' && touch f_nvcc_' + h + '.pdf ' +
        '&& touch d_nvcc_' + h + '.pdf ' +
        '&& touch f_all_' + h + '.pdf ' +
        '&& touch d_all_' + h + '.pdf ' +
        '&& touch e_all_' + h + '.pdf '
    )
    gcmd += (
        'psql flit -c \\"select createschmoo(' + str(run_num) + ',' +
        '\'{\\"f\\"}\',\'{\\"nvcc\\"}\',' + 
        '\'{\\"\\"}\',' +
        '\'' + h + '\',5,\'' + plot_dir + '/f_nvcc_' + h + '.pdf\')\\" & ' +
        'psql flit -c \\"select createschmoo(' + str(run_num) + ',' +
        '\'{\\"d\\"}\',\'{\\"nvcc\\"}\',' + 
        '\'{\\"\\"}\',' +
        '\'' + h + '\',5,\'' + plot_dir + '/d_nvcc_' + h + '.pdf\')\\" & ' +
        'psql flit -c \\"select createschmoo(' + str(run_num) + ',' +
        '\'{\\"f\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
        '\'{\\"-O1\\", \\"-O2\\", \\"-O3\\"}\',' +
        '\'' + h + '\',3,\'' + plot_dir + '/f_all_' + h + '.pdf\')\\" & ' +
        'psql flit -c \\"select createschmoo(' + str(run_num) + ',' +
        '\'{\\"d\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
        '\'{\\"-O1\\", \\"-O2\\", \\"-O3\\"}\',' +
        '\'' + h + '\',3,\'' + plot_dir + '/d_all_' + h + '.pdf\')\\" & ' +
        'psql flit -c \\"select createschmoo(' + str(run_num) + ',' +
        '\'{\\"e\\"}\',\'{\\"icpc\\", \\"g++\\", \\"clang++\\"}\',' + 
        '\'{\\"-O1\\", \\"-O2\\", \\"-O3\\"}\',' +
        '\'' + h + '\',3,\'' + plot_dir + '/e_all_' + h + '.pdf\')\\" & ' 
    )
pcmd += '&& chmod 777 * '

gcmd += ' wait'

#print('cmd line: ' + cmd)

print(check_output('sshpass -e ' + SSHS + db_host[0] + '@' + db_host[1] +
                    ' "' + pcmd + '"', env=new_env,
                   shell=True).decode("utf-8"))

print(check_output('sshpass -e ' + SSHS + db_host[0] + '@' + db_host[1] +
                    ' "' + gcmd + '"', env=new_env,
                   shell=True).decode("utf-8"))

print('Finished FLiT Run.  You may review the reports at: ' +
      db_host[0] + '@' + db_host[1] + ':' + plot_dir)
