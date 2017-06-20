#include "helper.h"

#include "QFPHelpers.hpp"

#include <iostream>
#include <iomanip>

/// RAII class for restoring iostream formats
class FmtRestore {
public:
  FmtRestore(std::ios& stream) : _stream(stream), _state(nullptr) {
    _state.copyfmt(_stream);
  }
  ~FmtRestore() { _stream.copyfmt(_state); }
private:
  std::ios& _stream;
  std::ios  _state;
};

void printTestVal(const std::string &funcName, float val) {
  FmtRestore restorer(std::cout);
  Q_UNUSED(restorer);

  auto intval = flit::as_int(val);
  std::cout << funcName << ":     0x"
            << std::hex << std::setw(8) << std::setfill('0') << intval
            << "  "
            << std::scientific << val
            << std::endl;
}

void printTestVal(const std::string &funcName, double val) {
  FmtRestore restorer(std::cout);
  Q_UNUSED(restorer);

  auto intval = flit::as_int(val);
  std::cout << funcName << ":     0x"
            << std::hex << std::setw(16) << std::setfill('0') << intval
            << "  "
            << std::scientific << val
            << std::endl;
}

void printTestVal(const std::string &funcName, long double val) {
  FmtRestore restorer(std::cout);
  Q_UNUSED(restorer);

  auto intval = flit::as_int(val);
  uint64_t lhalf = static_cast<uint64_t>((intval >> 64)) & 0xFFFFL;
  uint64_t rhalf = static_cast<uint64_t>(intval);

  std::cout << funcName << ":     0x"
            << std::hex << std::setw(4) << std::setfill('0') << lhalf
            << std::hex << std::setw(16) << std::setfill('0') << rhalf
            << "  "
            << std::scientific << val
            << std::endl;
}

namespace {

  auto generateSeed() {
    std::random_device seedGenerator;
    return seedGenerator();
  }

} // end of unnamed namespace

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

float randRealFloatGenerator() {
  static auto seed = generateSeed();
  static std::mt19937 generator32(seed);
  static std::uniform_real_distribution<float> distribution(
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max()
      );
  return distribution(generator32);
}

double randRealDoubleGenerator() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  static std::uniform_real_distribution<double> distribution(
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max()
      );
  return distribution(generator64);
}

long double randRealLongDoubleGenerator() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  static std::uniform_real_distribution<long double> distribution(
      std::numeric_limits<long double>::min(),
      std::numeric_limits<long double>::max()
      );
  return distribution(generator64);
}
