#include "helper.h"
#include "groundtruth.h"
#include "testbed.h"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

//TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

//template <typename T>
//using TruthType = std::pair<std::vector<T>, long double>;
//
//TruthType<float>
//runGroundtruth(const std::string &testName, std::function<float()> randGen) {
//  return 
//}


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
    auto testval = runTestbed(testName, inputs);
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
  runTest<float>("DistributivityOfMultiplication", 10, 1000000);

  return 0;
}

