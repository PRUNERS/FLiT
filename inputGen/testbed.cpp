#include "testbed.h"

#include <flit.h>

#include <iterator>

// Only store these locally because we want multiple compiled copies
namespace {
  template<typename T> long double
  runTestbed_impl(const std::string &testName,
                  const std::vector<T> &inputvals)
  {
    using flit::TestInput;
    using flit::Vector;

    auto test = flit::getTests()[testName]->get<T>();
    TestInput<T> input = test->getDefaultInput();
    input.vals = inputvals;
    auto scores = test->run(input);

    // Return only the first score.  Ignore the key
    return std::get<0>(scores.begin()->second);
  }
} // end of unnamed namespace


long double
runTestbed_float(const std::string &testName, const std::vector<float> &inputs) {
  return runTestbed_impl(testName, inputs);
}

long double
runTestbed_double(const std::string &testName, const std::vector<double> &inputs) {
  return runTestbed_impl(testName, inputs);
}

long double
runTestbed_longdouble(const std::string &testName, const std::vector<long double> &inputs) {
  return runTestbed_impl(testName, inputs);
}
