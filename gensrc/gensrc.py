#!/usr/bin/env python3
'''
This script generates random FLiT tests
'''

from environment import Environment, Variable
from expression import random_expression
from testcase import TestCase

import argparse
import math
import numpy as np
import os
import random
import sys


# TODO: write unit tests and put them somewhere else
def test_random_expression():
   print('Running test_random_expression()')
   env = Environment({
      'x': Variable('x', 'int'),
      'y': Variable('y', 'float'),
      'z': Variable('z', 'T'),
      })
   for i in range(10):
      print(random_expression(env, random.randint(0, 15), True))
   print()

def test_TestCase():
   print('Running test_TestCase()')
   tc = TestCase('TestCaseNumberOne', [6, 4, 2])
   print(tc)
   print()

def create_many_test_cases(directory, n):
    digits = math.ceil(math.log10(n + 1))
    name_template = 'RandomArithmeticTestCase_{0:0' + str(digits) + 'd}'
    for i in range(n):
        input_count = max(1, np.random.poisson(3))
        tc = TestCase(name_template.format(i+1), np.random.exponential(20, input_count))
        with open(os.path.join(directory, tc.name + '.cpp'), 'w') as outfile:
            outfile.write(str(tc))

def main(arguments):
   'Main logic here.  Returns 0 on success.'
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', '--outdir', default='.', help='Directory to output random tests')
   parser.add_argument('number', type=int, help='How many test cases to generate')
   args = parser.parse_args(arguments)
   #test_random_expression()
   #test_TestCase()
   create_many_test_cases(args.outdir, args.number)
   return 0

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))

