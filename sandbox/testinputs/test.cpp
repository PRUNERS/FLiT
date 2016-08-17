#include "test.h"
#include "functions.hpp"

#include <vector>

#define TST_DEFINE(funcName)                                        \
  float tst_##funcName(const std::vector<float> &in) {              \
    return funcName(in);                                            \
  }                                                                 \
  double tst_##funcName(const std::vector<double> &in) {            \
    return funcName(in);                                            \
  }                                                                 \
  long double tst_##funcName(const std::vector<long double> &in) {  \
    return funcName(in);                                            \
  }

TST_DEFINE(distribution)
