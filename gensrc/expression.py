# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

'Contains functions and classes for Expressions'

import random

class Expression(object):
   'An expression object'
   plus = '+'
   minus = '-'
   divide = '/'
   multiply = '*'

   ops = [
      plus,
      minus,
      divide,
      multiply,
      ]

   def __init__(self, parent=None):
      'initialize a leaf node'
      self.op = None
      self.left = None
      self.right = None
      self.parent = parent
      self.value = None

   def is_leaf(self):
      'True means this node is a leaf'
      return self.op == None

   def all_leaves(self):
      'Returns a list of all leaves'
      leaves = []
      if self.is_leaf():
         leaves.append(self)
      else:
         leaves.extend(self.left.all_leaves())
         leaves.extend(self.right.all_leaves())
      return leaves

   def choose_random_leaf(self):
      'Randomly choose and return a leaf node'
      if self.is_leaf():
         return self
      return random.choice(self.all_leaves())

   def expand(self):
      'Expands a leaf node'
      assert self.is_leaf()

      self.op = random.choice(self.ops)
      self.left = Expression(self)
      self.right = Expression(self)

   def __str__(self):
      'Convert to string'
      if self.is_leaf():
         return str(self.value)
      surround = False
      if self.parent is not None and self.parent.op in [self.multiply, self.divide]:
         surround = True
      retstr = ''
      if surround:
         retstr += '('
      retstr += str(self.left)
      retstr += ' ' + self.op + ' '
      retstr += str(self.right)
      if surround:
         retstr += ')'
      return retstr

   def __repr__(self):
      'Return a string representation of this expression'
      return 'Expression(' + str(self) + ')'

def random_expression(env, length, vars_only=False):
   '''
   Generates a random mathematical expression as a string
   '''
   # Populate the expression with operations
   expr = Expression()
   for i in range(length-1):
      leaf = expr.choose_random_leaf()
      leaf.expand()

   # Replace all of the leaf nodes with either variables or literals
   rand_var = lambda: random.choice(list(env.values()))
   #rand_flt = lambda: random.uniform(-1e9, 1e9)
   rand_flt = lambda: random.normalvariate(0, 5e4)
   rand_int = lambda: random.randint(-1e9, 1e9)

   randers = []
   if len(env) > 0:
      randers.append(rand_var)
   if not vars_only:
      randers.append(rand_flt)

   for leaf in expr.all_leaves():
      rander = random.choice(randers)
      leaf.value = rander()

   # Generate the string
   return expr
