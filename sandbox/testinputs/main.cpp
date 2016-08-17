#include "groundtruth.h"
#include "test.h"
#include "helper.h"
#include "functions.hpp"

#include <typeinfo>
#include <random>
#include <tuple>
#include <iostream>
#include <cmath>
#include <type_traits>

void printTestVal(const char* funcName, float val) {
  auto intval = reinterpret_float_as_int(val);
  printf("%s:     0x%08x  %e\n", funcName, intval, val);
}

void printTestVal(const char* funcName, double val) {
  auto intval = reinterpret_float_as_int(val);
  printf("%s:     0x%016lx  %e\n", funcName, intval, val);
}

void printTestVal(const char* funcName, long double val) {
  auto intval = reinterpret_float_as_int(val);
  printf("%s:     0x%04lx%016lx  %Le\n",
         funcName,
         static_cast<uint64_t>(intval >> 64),
         static_cast<uint64_t>(intval),
         val);
}

#define TESTRUN_DEFINE(funcName, inputCount, floatType)                          \
  template <typename T>                                                          \
  void testrun_##funcName(uint divergentCount, uint maxTries) {                  \
    std::vector<std::vector<T>> mismatchedInputs;                                \
    uint i;                                                                      \
    for (i = 0; mismatchedInputs.size() < divergentCount && i < maxTries; i++) { \
      auto inputs = generateRandomFloats<T>(inputCount, floatType);              \
      T truth = gt_##funcName(inputs);                                           \
      T testval = tst_##funcName(inputs);                                        \
      if (truth != testval) {                                                    \
        mismatchedInputs.push_back(inputs);                                      \
      }                                                                          \
    }                                                                            \
    printf("%s: %ld Diverging inputs found (%s) after %d iterations:\n",         \
           #funcName,                                                            \
           mismatchedInputs.size(),                                              \
           typeid(T).name(),                                                     \
           i                                                                     \
           );                                                                    \
    for (uint i = 0; i < mismatchedInputs.size(); i++) {                         \
      auto& input = mismatchedInputs[i];                                         \
      printf("%s:   Divergence #%d\n", #funcName, i + 1);                        \
      for (auto& val : input) {                                                  \
        printTestVal(#funcName, val);                                            \
      }                                                                          \
    }                                                                            \
    printf("\n");                                                                \
  }                                                                              \

#define TESTRUN(funcName, divergentCount, maxTries)                 \
  testrun_##funcName<float>(divergentCount, maxTries);              \
  testrun_##funcName<double>(divergentCount, maxTries);             \
  testrun_##funcName<long double>(divergentCount, maxTries);        \

TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

int main(void) {
  TESTRUN(distribution, 10, 1000000);

  return 0;
}

