from expression import random_expression;
from environment import Environment, Variable

import os

# Need to specify:
# - name: of the class
# - input_count: how many inputs the test will take
# - default_input: populate ti.vals vector.
# - vars_initialize: initialize scope variable for the test using ti.vals
# - cu_vars_initialize: initialize scope variables for the test in CUDA using tiList[idx].vals
# - func_body: test body that is shared between cuda and non-cuda.  Populate score1 and score2
template_string = '''
#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
{name}Kernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results) {{
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  decltype(results->s1) score1 = 0.0;
  decltype(results->s2) score2 = 0.0;

  {cu_vars_initialize}

  {{
    {func_body}
  }}

  results[idx].s1 = score1;
  results[idx].s2 = score2;
}}

template <typename T>
class {name} : public QFPTest::TestBase<T> {{
public:
  {name}(std::string id)
    : QFPTest::TestBase<T>(std::move(id)) {{}}

  virtual size_t getInputsPerRun() {{ return {input_count}; }}
  virtual QFPTest::TestInput<T> getDefaultInput() {{
    QFPTest::TestInput<T> ti;

    {default_input}

    return ti;
  }}

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() {{
    return {name}Kernel;
  }}

  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {{
    T score1 = 0.0;
    T score2 = 0.0;

    QFPHelpers::info_stream << id << ": Starting test with parameters" << std::endl;
    for (T val : ti.vals) {{
      QFPHelpers::info_stream << id << ":   " << val << std::endl;
    }}

    {vars_initialize}

    QFPHelpers::info_stream << id << ": After initializing variables" << std::endl;

    {func_body}

    QFPHelpers::info_stream << id << ": Ending test with values (" << score1 << ", " << score2 << ")" << std::endl;

    return {{score1, score2}};
  }}

protected:
  using QFPTest::TestBase<T>::id;
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
        'ti.vals.push_back({0});'.format(x) for x in default_input_vals]
    self.vars_initialize_lines = [
        'T in_{0} = ti.vals[{0}];'.format(i+1) for i in range(self.input_count)]
    self.cu_vars_initialize_lines = [
        'T in_{0} = tiList[idx].vals[{0}];'.format(i+1) for i in range(self.input_count)]

    # Create an environment for the function body
    env = Environment({
      #'score1': Variable('score1', 'T'),
      #'score2': Variable('score2', 'T'),
      })
    var_list = [Variable('in_{0}'.format(i+1), 'T') for i in range(self.input_count)]
    env.update(zip([x.name for x in var_list], var_list))

    # TODO: fill in the body with more logic
    self.func_body_lines = []
    for i in range(10):
      var = Variable('e{0}'.format(i+1), 'T')
      self.func_body_lines.append('{0} {1} = {2};'.format(var.type, var.name, random_expression(env, 3)))
      env[var.name] = var
    self.func_body_lines.append('score1 = {0};'.format(random_expression(env, 4, vars_only=True)))
    self.func_body_lines.append('score2 = {0};'.format(random_expression(env, 4, vars_only=True)))

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
      cu_vars_initialize='\n  '.join(self.cu_vars_initialize_lines),
      func_body='\n    '.join(self.func_body_lines),
      )

