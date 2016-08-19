#include "helper.h"

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
         static_cast<uint64_t>((intval >> 64)) & 0xFFFFL,
         static_cast<uint64_t>(intval),
         val);
}
