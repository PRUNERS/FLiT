#!/usr/bin/env python3

#this script takes to indices from the pc psql (qfp) database
#and invokes QD through the doDiffGdb.py script.
# IMPORTANT: this assumes that the pair of tests chosen is
# of the same type (psql(qfp):tests[index1].name ==
# psql(qfp):tests[index2].name)
# ALSO inter-host tests aren't handled yet.  This assumes
# you are on the correct host for both tests.

import sys
from subprocess import check_output
from os import chdir, environ
from pg import DB #PyGreSQL


# handle usage

def usage():
    print(sys.argv[0] + ' [test index 1] [test index 2]')

if not len(sys.argv) == 3:
    usage()
    exit(1)

#extract params

test1 = sys.argv[1]
test2 = sys.argv[2]
tpath1 = 'test1'
tpath2 = 'test2'

# read data from db
# these are the params we need for build
# comp1, comp2, flags1, flags2
# these are the params we need for doDiffGdb.py:
# tname prec1 prec2 sort1 sort2

db = DB(dbname='qfp', host='localhost', user='qfp', passwd='qfp123')
q1 = db.query('select * from tests where index  = ' + test1)
q2 = db.query('select * from tests where index  = ' + test2)

d1 = q1.dictresult()[0]
print('queried for test 1, found? ' + str(len(d1) > 0))
comp1 = d1['compiler']
flags1 = d1['switches']
prec1 = d1['precision']
tname = d1['name']
sort1 = d1['sort']

d2 = q2.dictresult()[0];
print('queried for test 2, found? ' + str(len(d2) > 0))
comp2 = d2['compiler']
flags2 = d2['switches']
prec2 = d2['precision']
sort2 = d2['sort']
tname2 = d2['name']

# print(tname + ":" + tname2)
# assert tname2 == tname


#build required tests

chdir('perpVects')

print('building test1')
print('with CC = ' + comp1 + ', FFLAGS = ' + flags1 + ', TARGET = ' +
      tpath1)
environ['CC'] = comp1
environ['FFLAGS'] = flags1
environ['TARGET'] = tpath1
#DEBUG uncomment next line
#print(check_output(['make', '-f', 'MakefileDev', 'clean']))
print(check_output(['make', '-j', '8', '-f', 'MakefileDev']))

print('building test2')
print('with CC = ' + comp2 + ', FFLAGS = ' + flags2 + ', TARGET = ' +
      tpath2)
environ['CC'] = comp2
environ['FFLAGS'] = flags2
environ['TARGET'] = tpath2
#DEBUG uncomment next line
#print(check_output(['make', '-f', 'MakefileDev', 'clean']))
print(check_output(['make', '-j', '8', '-f', 'MakefileDev']))

#invoke doDiffGdb

chdir('..')
print(check_output(['gdb/doDiffGdb.py', tpath1, tpath2, tname, prec1,
                    sort1, tname2, prec2, sort2]))

