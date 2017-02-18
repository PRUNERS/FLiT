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

   >>> env = Environment({'x'; Variable('x', 'int'})
   >>> random_expression(env, 3)
   Expression((x + 3.25124) * x)
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
   var = rand_var()
   return expr
