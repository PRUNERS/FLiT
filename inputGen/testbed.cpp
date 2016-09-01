#include "testbed.h"

#include "DistributivityOfMultiplication.hpp"
#include "DoHariGSBasic.hpp"
#include "DoHariGSImproved.hpp"
#include "DoMatrixMultSanity.hpp"
#include "DoOrthoPerturbTest.hpp"
#include "DoSimpleRotate90.hpp"
#include "DoSkewSymCPRotationTest.hpp"
#include "RotateAndUnrotate.hpp"
#include "RotateFullCircle.hpp"
#include "TrianglePHeron.hpp"
#include "TrianglePSylv.hpp"

#include "testBase.hpp"
#include "QFPHelpers.hpp"

// Only store these locally because we want multiple compiled copies
namespace {
  std::map<std::string, QFPTest::TestFactory*> alltests;
  void registerGtTest(std::string name, QFPTest::TestFactory* factory) {
    alltests[name] = factory;
  }

  INTERNAL_REGISTER_TYPE(DistributivityOfMultiplication, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoHariGSBasic, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoHariGSImproved, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoMatrixMultSanity, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoOrthoPerturbTest, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoSimpleRotate90, registerGtTest)
  INTERNAL_REGISTER_TYPE(DoSkewSymCPRotationTest, registerGtTest)
  INTERNAL_REGISTER_TYPE(RotateAndUnrotate, registerGtTest)
  INTERNAL_REGISTER_TYPE(RotateFullCircle, registerGtTest)
  INTERNAL_REGISTER_TYPE(TrianglePHeron, registerGtTest)
  INTERNAL_REGISTER_TYPE(TrianglePSylv, registerGtTest)

  template<typename T> long double
  runTestbed_impl(const std::string &testName,
                  const std::vector<T> &inputvals)
  {
    using QFPTest::TestInput;
    using QFPHelpers::Vector;

    auto test = alltests[testName]->get<T>();
    TestInput<T> input = test->getDefaultInput();
    input.vals = inputvals;
    auto scores = test->run(input);

    // Return only the first score.  Ignore the key
    return std::get<0>(scores.begin()->second);
  }
}


long double
runTestbed(const std::string &testName, const std::vector<float> &inputs) {
  return runTestbed_impl(testName, inputs);
}

long double
runTestbed(const std::string &testName, const std::vector<double> &inputs) {
  return runTestbed_impl(testName, inputs);
}

long double
runTestbed(const std::string &testName, const std::vector<long double> &inputs) {
  return runTestbed_impl(testName, inputs);
}
