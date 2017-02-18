#!/usr/bin/env python3
'''
This script generates random FLiT tests
'''

from environment import Environment, Variable
from expression import random_expression
from testcase import TestCase

import argparse
import random
import sys


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

def main(arguments):
   'Main logic here.  Returns 0 on success.'
   parser = argparse.ArgumentParser()
   args = parser.parse_args(arguments)
   #test_random_expression()
   test_TestCase()
   return 0

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))

