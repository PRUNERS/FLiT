#include "groundtruth.h"
#include "functions.hpp"

#include <vector>

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

GT_DEFINE(distribution)
