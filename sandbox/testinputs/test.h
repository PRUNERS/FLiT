#ifndef TEST_H
#define TEST_H

#include "functions.hpp"
#include <vector>

#define TST_DECLARE(funcName)                                   \
  float tst_##funcName(const std::vector<float>&);              \
  double tst_##funcName(const std::vector<double>&);            \
  long double tst_##funcName(const std::vector<long double>&);  \

TST_DECLARE(distribution)

#endif // TEST_H
