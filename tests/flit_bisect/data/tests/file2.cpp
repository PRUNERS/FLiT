#include "file2.h"

#include <flit.h>

#include <string>

int file2_func1_PROBLEM() {
  if (std::string(FLIT_OPTL) == "-O3") {
    return 1 + 7;  // variability introduced = 7
  } else {
    return 1;
  }
}

int file2_func2() { return 2; }
int file2_func3() { return 3; }
int file2_func4() { return 4; }
int file2_func5() { return 5; }

int file2_all() {
  return file2_func1_PROBLEM()
       + file2_func2()
       + file2_func3()
       + file2_func4()
       + file2_func5();
}
