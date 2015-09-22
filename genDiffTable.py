#!/usr/bin/python3

import glob
#from subprocess import call
import os

os.chdir("results")
os.system("rm *_out*")
os.system("find . -name '*.tgz' -exec tar -zxf {} \;")

filelist = glob.glob("*")

item = filelist.pop()

diffMap = {}
while len(filelist) != 0:
    for f in filelist:
        diffMap[f].append({f, os.system("diff " + f + " " + item)})
    item = filelist.pop()

#OK, built the map of diffs -- now output to webpage table

wstr = "<table>"

filelist = glob.glob("./results/*")

wstr = wstr + "<tr>"
for i in filelist:
    wstr = wstr + "<td>" + f + "</td>"
wstr = wstr + "</tr>"

#header built, now add rows
skip = 0
for x in filelist:
    for s in range(0, skip):
        wstr = wstr + "<td/>"
        for f,b in diffMap[x]:
            wstr = wstr + "<td>" + b + "</td>"

print (wstr)
