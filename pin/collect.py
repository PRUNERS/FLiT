#!/usr/bin/env python3

#this script runs the pin opcodemix to collect QC results.
#It works in the results directory on a host that ran QC (wihtout deleting output)

import glob
import sys
import os
import subprocess

if not os.path.exists('opcodes'):
    os.makedirs('opcodes')

for f in glob.glob('*'):
    if f[:-4] == 'out_':
        continue
    splt = f.split('_')
    if len(splt) != 3:
        continue
    compiler = splt[0]
    host = splt[1]
    flags = splt[2]
    for p in ['f', 'd', 'e']:
        os.environ['PRECISION'] = p
        for t in ['DistributivityOfMultiplication',
                  'RotateAndUnrotate',
                  'DoSkewSymCPRotationTest',
                  'DoHariGSBasic',
                  'DoHariGSImproved',
                  'DoSimpleRotate90',
                  'TrianglePHeron',
                  'DoMatrixMultSanity',
                  'DoOrthoPerturbTest',
                  'TrianglePSylv',
                  'RotateFullCircle',
                  'DistributivityOfMultiplication']:

            os.environ['TEST'] = t
            print(subprocess.check_output(['../pin/pin/pin', '-t',
                                           '../pin/pin/source/tools/SimpleExamples/obj-intel64/opcodemix.so',
                                           '-o', 'opcodes/' + f + '_' + p + '_' + t + '.pin', '--', './' +
                                f]))
