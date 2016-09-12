#include "helper.h"
#include "groundtruth.h"
#include "testBase.hpp"
//#include "testbed.h"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include <dlfcn.h>    // For dlopen(), dlsym() and dlclose()

//TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

template<typename T> const char* testbedSymbolName();
template<> const char* testbedSymbolName<float>() { return "runTestbed_float"; }
template<> const char* testbedSymbolName<double>() { return "runTestbed_double"; }
template<> const char* testbedSymbolName<long double>() { return "runTestbed_longdouble"; }

template<typename T> long double
runTestbed(const std::string &testName, const std::vector<T> &inputs) {
  const char* lib = "./testbed.so";
  const char* symb = testbedSymbolName<T>();

  using TestBedFnc = long double(const std::string&, const std::vector<T>&);
  auto handle = dlopen(lib,
      RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  auto testbed = reinterpret_cast<TestBedFnc*>(dlsym(handle, symb));
  if (testbed == nullptr) {
    std::cerr << "Error: could not find symbol " << symb << " from " << lib << std::endl;
    std::exit(1);
  }
  auto retval = testbed(testName, inputs);

  dlclose(handle);
  
  return retval;
}


template<typename T>
void runTest(std::string testName, uint divergentCount, uint maxTries) {
  const std::function<T()> randGen = []() {
    return generateRandomFloat<T>(RandomFloatType::Positive);
  };

  std::cout << testName << "(" << typeid(T).name() << "): "
            << "looking for " << divergentCount << " divergences, "
            << "max_tries = " << maxTries
            << std::endl;

  uint mismatchIdx;
  std::vector<std::vector<T>> mismatches;
  for (mismatchIdx = 0;
       mismatches.size() < divergentCount && mismatchIdx < maxTries;
       mismatchIdx++)
  {
    auto truthRun = runGroundtruth(testName, randGen);
    auto inputs = truthRun.first;
    auto truthval = truthRun.second;
    auto testval = runTestbed(testName, inputs);
    if (truthval != testval && !(std::isnan(truthval) && std::isnan(testval))) {
      mismatches.push_back(inputs);

      std::cout << testName << ":   Divergent outputs #" << mismatches.size() << std::endl;
      printTestVal(testName, truthval);
      printTestVal(testName, testval);
      std::cout << testName << ":   Divergent inputs #" << mismatches.size() << std::endl;
      for (auto& val : inputs) {
        printTestVal(testName, val);
      }
    }
  }
  std::cout << testName << "(" << typeid(T).name() << "): "
            << mismatches.size() << " Diverging inputs found "
            << "after " << mismatchIdx << " iterations:"
            << std::endl
            << std::endl;
}

void runAllPrecisions(const std::string testName, uint divergentCount, uint maxTries) {
  runTest<float>(testName, divergentCount, maxTries);
  runTest<double>(testName, divergentCount, maxTries);
  runTest<long double>(testName, divergentCount, maxTries);
}

std::vector<std::string> getTestNames() {
  std::vector<std::string> retval;
  for (auto entry : QFPTest::getTests()) {
    retval.push_back(entry.first);
  }
  return retval;
}


int main(void) {
  for (auto testname : getTestNames()) {
    runTest<float>(testname, 10, 1000);
  }
  // Sanity check: should not find any differences no matter the optimizations
  //runAllPrecisions("DoMatrixMultSanity", 10, 1000);

  //runTest<float>("DistributivityOfMultiplication", 10, 100000);
  //runTest<double>("DistributivityOfMultiplication", 10, 100000);
  //runTest<long double>("DistributivityOfMultiplication", 10, 100000);
  //runAllPrecisions("DistributivityOfMultiplication", 10, 10000);

  //runAllPrecisions("DoHariGSBasic", 10, 100000);
  //QFPHelpers::info_stream.show();

  return 0;
}

