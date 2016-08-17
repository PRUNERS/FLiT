#ifndef MACROS_H
#define MACROS_H

#include "helper.h"

#include <vector>
#include <cstdio>

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

#define GT_DECLARE(funcName)                                   \
  float gt_##funcName(const std::vector<float>&);              \
  double gt_##funcName(const std::vector<double>&);            \
  long double gt_##funcName(const std::vector<long double>&);  \

#define TST_DECLARE(funcName)                                   \
  float tst_##funcName(const std::vector<float>&);              \
  double tst_##funcName(const std::vector<double>&);            \
  long double tst_##funcName(const std::vector<long double>&);  \

#define TST_DEFINE(funcName)                                        \
  float tst_##funcName(const std::vector<float> &in) {              \
    return funcName(in);                                            \
  }                                                                 \
  double tst_##funcName(const std::vector<double> &in) {            \
    return funcName(in);                                            \
  }                                                                 \
  long double tst_##funcName(const std::vector<long double> &in) {  \
    return funcName(in);                                            \
  }                                                                 \

#define GT_DEFINE(funcName)                                         \
  float gt_##funcName(const std::vector<float> &in) {               \
    return funcName(in);                                            \
  }                                                                 \
  double gt_##funcName(const std::vector<double> &in) {             \
    return funcName(in);                                            \
  }                                                                 \
  long double gt_##funcName(const std::vector<long double> &in) {   \
    return funcName(in);                                            \
  }                                                                 \

#endif // MACROS_H
