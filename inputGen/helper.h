#ifndef HELPER_H
#define HELPER_H

#include <flit.h>

#include <type_traits>
#include <random>
#include <stdexcept>


void printTestVal(const std::string &funcName, float val);
void printTestVal(const std::string &funcName, double val);
void printTestVal(const std::string &funcName, long double val);

uint32_t randGenerator32();
uint64_t randGenerator64();
unsigned __int128 randGenerator128();
float randRealFloatGenerator();
double randRealDoubleGenerator();
long double randRealLongDoubleGenerator();


enum class RandType {
  UniformFP,    // uniform on the space of Floating-Point(FP) numbers
  UniformReals, // uniform on the real number line, then projected to FP
};

template <typename F>
F generateFloatBits(RandType type = RandType::UniformFP);

template<> inline
float generateFloatBits<float>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator32());
    case RandType::UniformReals:
      return randRealFloatGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

template<> inline
double generateFloatBits<double>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator64());
    case RandType::UniformReals:
      return randRealDoubleGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

template<> inline
long double generateFloatBits<long double>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator128());
    case RandType::UniformReals:
      return randRealLongDoubleGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

enum class RandomFloatType {
  Positive,
  Negative,
  Any,
};

template <typename T>
T generateRandomFloat(RandomFloatType fType = RandomFloatType::Any,
                      RandType rType = RandType::UniformFP) {
  static_assert(
      std::is_floating_point<T>::value,
      "generateRandomFloats() can only be used with floating point types"
      );

  // Generate a random floating point number
  T val;
  do {
    val = generateFloatBits<T>(rType);
  } while (std::isnan(val));

  // Convert the values based on the desired qualities
  if (fType == RandomFloatType::Positive) {
    val = std::abs(val);
  } else if (fType == RandomFloatType::Negative) {
    val = -std::abs(val);
  } else if (fType == RandomFloatType::Any) {
    // Do nothing
  } else {
    throw std::runtime_error("Unsupported RandomFloatType passed in");
  }

  return val;
}

#endif // HELPER_H
