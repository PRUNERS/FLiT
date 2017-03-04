'''
The Environment class along with things needed to define an environment
'''

from collections import namedtuple

Variable = namedtuple('Variable', 'name, type')
class Variable(object):
   'Represents a variable with a certain type'

   def __init__(self, name, vartype):
      'initialize this'
      self.name = name
      self.type = vartype
   def __str__(self):
      'return this object as a string'
      return self.name
   def __repr__(self):
      'return this object as a string, but retaining the type'
      return 'Variable({0}, {1})'.format(self.name.__repr__(), self.type.__repr__())

class Environment(dict):
   'Represents an environment of variables'

   def __init__(self, parent=None):
      'Initialize an environment.  Copies from the parent if not None.'
      self.parent = parent
      if parent is not None:
         self.update(parent)

empty_env = Environment()

