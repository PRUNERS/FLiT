#!/usr/bin/python3

import glob
from subprocess import call
import os

os.chdir("results")
os.system("rm *_out*")
os.system("find . -name '*.tgz' -exec tar -zxf {} \;")

filelist = sorted(glob.glob("*_out*"))
#print(filelist)

diffMap = []
count = 0
for item in filelist:
    diffMap.append([])
    for f in filelist:
        diffMap[count].append([f, call(["diff", f, item], stdout=open(os.devnull, 'wb'))])
    count = count + 1
        # diffMap[item].append([f, os.system("diff " + f + " " + item)])

#print (diffMap)
#OK, built the map of diffs -- now output to webpage table

wstr = "<table border=\"1\"><tr><td></td>"

filelist = sorted(glob.glob("*_out*"))
for f in filelist:
    wstr = wstr + "<td>" + f + "</td>"

wstr = wstr + "</tr>"

#header built, now add rows
skip = 1
count = 0
for x in filelist:
    wstr = wstr + "<tr><td>" + x + "</td>"
    # for s in range(0, skip):
    #     wstr = wstr + "<td>nocheck</td>"
    # skip = skip + 1
    for f,r in diffMap[count]:
        if r != 0:
            res = "diff"
        else:
            res = "same"
        wstr = wstr + "<td>" + res + "</td>"
    wstr = wstr + "</tr>"
    count = count + 1 
wstr = wstr + "</table>"
f = open('difftable', 'w')

f.write(wstr)

#print (wstr)
