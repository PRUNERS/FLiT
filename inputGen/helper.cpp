#include "helper.h"

#include "QFPHelpers.hpp"

void printTestVal(const std::string &funcName, float val) {
  auto intval = QFPHelpers::as_int(val);
  printf("%s:     0x%08x  %e\n", funcName.c_str(), intval, val);
}

void printTestVal(const std::string &funcName, double val) {
  auto intval = QFPHelpers::as_int(val);
  printf("%s:     0x%016lx  %e\n", funcName.c_str(), intval, val);
}

void printTestVal(const std::string &funcName, long double val) {
  auto intval = QFPHelpers::as_int(val);
  printf("%s:     0x%04lx%016lx  %Le\n",
         funcName.c_str(),
         static_cast<uint64_t>((intval >> 64)) & 0xFFFFL,
         static_cast<uint64_t>(intval),
         val);
}

namespace {

  auto generateSeed() {
    std::random_device seedGenerator;
    return seedGenerator();
  }

}

uint32_t randGenerator32() {
  static auto seed = generateSeed();
  static std::mt19937 generator32(seed);
  return generator32();
}

uint64_t randGenerator64() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  return generator64();
}

unsigned __int128 randGenerator128() {
  static auto generator128 = []() {
    unsigned __int128 val = randGenerator64();
    val = val << 64;
    val += randGenerator64();
    return val;
  };
  return generator128();
}
