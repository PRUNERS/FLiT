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

from expression import random_expression;
from environment import Environment, Variable

import os

# Need to specify:
# - name: of the class
# - input_count: how many inputs the test will take
# - default_input: populate ti.vals vector.
# - vars_initialize: initialize scope variable for the test using ti.vals
#   tiList[idx].vals
# - func_body: test body
template_string = '''
#include "flit.h"

template <typename T>
class {name} : public flit::TestBase<T> {{
public:
  {name}(std::string id)
    : flit::TestBase<T>(std::move(id)) {{}}

  virtual size_t getInputsPerRun() override {{ return {input_count}; }}
  virtual std::vector<T> getDefaultInput() override {{
    std::vector<T> ti;

    {default_input}

    return ti;
  }}

protected:
  virtual
  flit::Variant run_impl(const std::vector<T>& ti) override {{
    T score = 0.0;

    flit::info_stream << id << ": Starting test with parameters" << std::endl;
    for (T val : ti) {{
      flit::info_stream << id << ":   " << val << std::endl;
    }}

    {vars_initialize}

    flit::info_stream << id << ": After initializing variables" << std::endl;

    {func_body}

    flit::info_stream << id << ": Ending test with value (" << score << ")" << std::endl;

    return score;
  }}

protected:
  using flit::TestBase<T>::id;
}};

REGISTER_TYPE({name})
'''

class TestCase(object):
  def __init__(self, name, default_input_vals):
    '''
    Initialize the test case
    @param name: the name of the test class.  Also names the file {name}.cpp
    @param default_input_vals: the default values passed as parameters to the test
    '''
    self.name = name
    self.default_input_vals = default_input_vals
    self.input_count = len(default_input_vals)

    # setup the test
    self.default_input_lines = [
        'ti.push_back({0});'.format(x) for x in default_input_vals]
    self.vars_initialize_lines = [
        'T in_{0} = ti[{0}];'.format(i+1) for i in range(self.input_count)]

    # Create an environment for the function body
    env = Environment({
      })
    var_list = [Variable('in_{0}'.format(i+1), 'T') for i in range(self.input_count)]
    env.update(zip([x.name for x in var_list], var_list))

    # TODO: fill in the body with more logic
    self.func_body_lines = []
    for i in range(10):
      var = Variable('e{0}'.format(i+1), 'T')
      self.func_body_lines.append('{0} {1} = {2};'.format(var.type, var.name,
                                  random_expression(env, 3)))
      env[var.name] = var
    self.func_body_lines.append('score = {0};'.format(random_expression(env, 4,
                                vars_only=True)))

  def write(self, directory='.'):
    '''
    Writes this test case to the file {self.name}.cpp into the specified
    directory
    '''
    with open(os.path.join(directory, self.name + '.cpp'), 'w') as outfile:
      outfile.write(str(self))

  def __str__(self):
    'outputs this object as a string.  It outputs the c++ file contents'
    return template_string.format(
      name=self.name,
      input_count=self.input_count,
      default_input='\n    '.join(self.default_input_lines),
      vars_initialize='\n    '.join(self.vars_initialize_lines),
      func_body='\n    '.join(self.func_body_lines),
      )

