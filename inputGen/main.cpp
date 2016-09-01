#include "helper.h"
#include "groundtruth.h"
//#include "testbed.h"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include <dlfcn.h>    // For dlopen(), dlsym() and dlclose()

//TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

long double
runTestbed(const char* testName, float* vals, int valLength) {
  (void)testName;
  (void)vals;
  (void)valLength;

  const char* lib = "./testbed.so";
  const char* symb = "runTestbed_float";

  using TestBedFnc = long double(const char*, float*, int);
  auto handle = dlopen(lib,
      RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  auto testbed = reinterpret_cast<TestBedFnc*>(dlsym(handle, symb));
  if (testbed == nullptr) {
    std::cerr << "Error: could not find symbol " << symb << " from " << lib << std::endl;
    std::exit(1);
  }
  auto retval = testbed(testName, vals, valLength);

  dlclose(handle);
  
  return retval;
}


template<typename T>
void runTest(std::string testName, uint divergentCount, uint maxTries) {
  const std::function<T()> randGen = []() {
    return generateRandomFloat<float>(RandomFloatType::Positive);
  };
  uint mismatchIdx;
  std::vector<std::vector<T>> mismatches;
  for (mismatchIdx = 0;
       mismatches.size() < divergentCount && mismatchIdx < maxTries;
       mismatchIdx++)
  {
    auto truthRun = runGroundtruth(testName, randGen);
    auto inputs = truthRun.first;
    auto truthval = truthRun.second;
    auto testval = runTestbed(testName.c_str(), inputs.data(), inputs.size());
    if (truthval != testval) {
      mismatches.push_back(inputs);
    }
  }
  std::cout << testName << ": "
            << mismatches.size() << " Diverging inputs found "
            << "(" << typeid(T).name() << ") "
            << "after " << mismatchIdx << " iterations:"
            << std::endl;
  for (uint i = 0; i < mismatches.size(); i++) {
    auto& input = mismatches[i];
    std::cout << testName << ":   Divergence #" << i + 1 << std::endl;
    for (auto& val : input) {
      printTestVal(testName, val);
    }
  }
  printf("\n");
}


int main(void) {
  runTest<float>("DistributivityOfMultiplication", 10, 100000);
  //runTest<double>("DistributivityOfMultiplication", 10, 100000);
  //runTest<long double>("DistributivityOfMultiplication", 10, 100000);

  return 0;
}

