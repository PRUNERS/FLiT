#include "functions.hpp"
#include "groundtruth.h"
#include "helper.h"
#include "macros.h"
#include "test.h"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

int main(void) {
  TESTRUN(distribution, 10, 1000000);

  return 0;
}

