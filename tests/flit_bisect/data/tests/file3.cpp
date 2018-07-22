#include "file1.h"

#include <flit.h>

#include <string>

int file3_func1() { return 1; }

int file3_func2_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 2 + 1;  // variability introduced = 1
  } else {
    return 2;
  }
}

int file3_func3() { return 3; }
int file3_func4() { return 4; }

int file3_func5_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 5 + 3;  // variability introduced = 3
  } else {
    return 5;
  }
}

int file3_all() {
  return file3_func1()
       + file3_func2_PROBLEM()
       + file3_func3()
       + file3_func4()
       + file3_func5_PROBLEM();
}
