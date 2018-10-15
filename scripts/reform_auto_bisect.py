#!/usr/bin/env python3

import csv
import glob
import re
import sys

writer = csv.writer(sys.stdout)
writer.writerow([
    'bisectnum',
    'bisectcount',
    'compiler',
    'optl',
    'switches',
    'precision',
    'testcase',
    'type',
    'name',
    'return',
    ])

for fn in glob.glob("output/test_*/bisect-*/bisect.log"):
    m = re.match("output/test_([0-9]*)/bisect-([0-9]*)/bisect.log", fn)
    testcase = "ex{}_Test".format(m.group(1))
    bisect_num = m.group(2)
    bisect_count = len(glob.glob('output/test_{}/bisect-{}/bisect-make-*.mk'.format(m.group(1), m.group(2))))

    compiler = None
    optl = None
    switches = None
    precision = "d"

    libs = None
    srcs = None
    syms = None

    status = 1

    INTRO = 1
    LIB_START = 2
    LIB = 3
    SRC_START = 4
    SRC = 5
    SYM_START = 6
    SYM = 7
    state = INTRO

    with open(fn, "r") as f:
        for line in f:

            if state == INTRO:
                # ...: Starting the bisect procedure
                # ...:   trouble compiler:           "g++"
                # ...:   trouble optimization level: "-O2"
                # ...:   trouble switches:           "-mavx2 -mfma"
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
                    state = LIB_START
                    continue

                continue


            if state == LIB_START:
                if "BAD STATIC LIBRARIES:" in line:
                    state = LIB
                    libs = list()
                    continue

                if "ALL VARIABILITY INCUDING SOURCE FILE(S):" in line:
                    state = SRC
                    libs = list()
                    srcs = list()
                    continue

                continue


            if state == LIB:
                msg = "May not be able to search further, because of intel optimizations"
                if line.count("None") >= 1 or line.count(msg) >= 1:
                    state = SRC_START
                else:
                    lib = line.split()[-1]
                    libs.append(lib)

                continue


            if state == SRC_START:
                if "ALL VARIABILITY INCUDING SOURCE FILE(S):" in line:
                    state = SRC
                    srcs = list()
                    continue

                continue


            if state == SRC:
                msg = "Searching for differing symbols in"
                if line.count("None") >= 1 or line.count(msg) >= 1:
                    state = SYM_START
                else:
                    assert(line != "(score")
                    src = line[line.index("bisect:") + 7 : line.index("(score")]
                    src = src.strip()
                    srcs.append(src)

                continue


            if state == SYM_START:
                if "ALL VARIABILITY INCUDING SYMBOLS:" in line:
                    status = 0
                    state = SYM
                    syms = list()
                    continue

                continue


            if state == SYM:
                if "None" in line:
                    break
                m = re.search(r".*\..*:[0-9]* *(.*) *\(score", line)
                assert(m != None)
                sym = m.group(1)
                syms.append(sym)

                continue

            assert(0)

    # bisectnum,compiler,optl,switches,precision,testcase,type,name,return
    completed = list()
    if libs != None:
        completed.append("lib")
        for lib in libs:
            writer.writerow([
                bisect_num,
                bisect_count,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "lib",
                lib,
                status,
                ])

    if srcs != None:
        completed.append("src")
        for source in srcs:
            writer.writerow([
                bisect_num,
                bisect_count,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "src",
                source,
                status,
                ])

    if syms != None:
        completed.append("sym")
        for sym in syms:
            writer.writerow([
                bisect_num,
                bisect_count,
                compiler,
                optl,
                switches,
                precision,
                testcase,
                "sym",
                sym,
                status,
                ])

    writer.writerow([
        bisect_num,
        bisect_count,
        compiler,
        optl,
        switches,
        precision,
        testcase,
        "completed",
        ','.join(completed),
        status,
        ])
