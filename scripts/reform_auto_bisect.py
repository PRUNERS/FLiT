#!/usr/bin/env python3

import glob
import re

for fn in glob.glob("output/test_*/bisect-*/bisect.log"):

    m = re.match("output/test_([0-9]*)/bisect-([0-9]*)/bisect.log", fn)
    testcase = "ex{}_Test".format(m.group())
    bisect_num = m.group(2)

    compiler = None
    optl = None
    switches = None
    precision = "d"

    libs = None
    srcs = None
    syms = None

    status = 0

    look_for_libs = False
    with open(fn, "r") as f:
        for line in f:
            assert(line != None)
            #2018-10-03 14:23:56,052 bisect: Starting the bisect procedure
            #2018-10-03 14:23:56,052 bisect:   trouble compiler:           "g++"
            #2018-10-03 14:23:56,052 bisect:   trouble optimization level: "-O2"
            #2018-10-03 14:23:56,053 bisect:   trouble switches:           "-mavx2 -mfma"
            m = re.search(r"trouble compiler:           \"(.*)\"", line)
            if m != None:
                compiler = m.group(1)
                continue

            m = re.search(r"trouble optimization level: \"(.*)\"", line)
            if m != None:
                optl = m.group(1)
                continue

            m = re.search(r"trouble switches:           \"(.*)\"", line)
            if m != None:
                switches = m.group(1)
                if compiler == "icpc":
                    switches = switches.split(maxsplit=1)[1]
                    optl = switches.split(maxsplit=1)[0]
                    try:
                        switches = switches.split(maxsplit=1)[1]
                    except IndexError:
                        switches = ""
                continue

            if line.count("BAD STATIC LIBRARIES:") >= 1:
                look_for_libs = True
                libs = list()
                continue

            if look_for_libs:
                if line.count("None") >= 1 or line.count("May not be able to search further, because of intel optimizations") >= 1:
                    look_for_libs = False
                else:
                    lib = line.split()[-1]
                    libs.append(lib)
                continue

            # Found differing source file /uufs/chpc.utah.edu/common/home/u0692013/FLIT/FLiT_mfem/mfem/linalg/sparsemat.cpp: score 3.452641112552743e-14
            m = re.search("Found differing source file (.*): score", line)
            if m != None:
                if srcs == None:
                    srcs = list()
                srcs.append(m.group(1))
                continue

            # Found differing symbol on line 493 -- mfem::SparseMatrix::AddMult(mfem::Vector const&, mfem::Vector&, double) const (score 3.452641112552743e-14)
            m = re.search("Found differing symbol on line [0-9]* -- (.*) \(score", line)
            if m != None:
                if syms == None:
                    syms = list()
                syms.append(m.group(1))
                continue


            if line.count("cannot continue") >= 1:
                status = 1
                if syms != None:
                    syms = None
                    break
                if srcs != None:
                    srcs = None
                    break
                if libs != None:
                    libs = None
                    break


    # bisectnum,compiler,optl,switches,precision,testcase,type,name,return
    completed = list()
    if libs != None:
        completed.append("lib")
        for lib in libs:
            print('{},{},{},"{}",{},{},{},"{}",{}'.format(
                bisect_num,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "lib",
                lib,
                status))

    if srcs != None:
        completed.append("src")
        for src in srcs:
            print('{},{},{},"{}",{},{},{},"{}",{}'.format(
                bisect_num,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "src",
                src,
                status))

    if syms != None:
        completed.append("sym")
        for sym in syms:
            print('{},{},{},"{}",{},{},{},"{}",{}'.format(
                bisect_num,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "sym",
                sym,
                status))

    print('{},{},{},"{}",{},{},{},"{}",{}'.format(
        bisect_num,
        compiler,
        optl,
        switches,
        precision,
        testcase,
        "completed",
        ','.join(completed),
        status))
