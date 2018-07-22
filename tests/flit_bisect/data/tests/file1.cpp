#include "file1.h"

#include <flit.h>

#include <string>

int file1_func1() { return 1; }

int file1_func2_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 2 + 5;  // variability introduced = 5
  } else {
    return 2;
  }
}

int file1_func3_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 3 + 2;  // variability introduced = 2
  } else {
    return 3;
  }
}

int file1_func4_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 4 + 3;  // variability introduced = 3
  } else {
    return 4;
  }
}

int file1_func5() { return 5; }

int file1_all() {
  return file1_func1()
       + file1_func2_PROBLEM()
       + file1_func3_PROBLEM()
       + file1_func4_PROBLEM()
       + file1_func5();
}
