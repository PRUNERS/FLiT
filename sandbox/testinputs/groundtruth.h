#ifndef GROUNDTRUTH_H
#define GROUNDTRUTH_H

#include "functions.hpp"
#include <vector>

#define GT_DECLARE(funcName)                                   \
  float gt_##funcName(const std::vector<float>&);              \
  double gt_##funcName(const std::vector<double>&);            \
  long double gt_##funcName(const std::vector<long double>&);  \

GT_DECLARE(distribution)

#endif // GROUNDTRUTH_H
