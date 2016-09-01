#ifndef HELPER_H
#define HELPER_H

#include "QFPHelpers.hpp"

#include <type_traits>
#include <random>
#include <stdexcept>


void printTestVal(const std::string &funcName, float val);
void printTestVal(const std::string &funcName, double val);
void printTestVal(const std::string &funcName, long double val);

uint32_t randGenerator32();
uint64_t randGenerator64();
unsigned __int128 randGenerator128();


template <typename F>
F generateFloatBits();

template<> inline
float generateFloatBits<float>() {
  return QFPHelpers::as_float(randGenerator32());
}

template<> inline
double generateFloatBits<double>() {
  return QFPHelpers::as_float(randGenerator64());
}

template<> inline
long double generateFloatBits<long double>() {
  return QFPHelpers::as_float(randGenerator128());
}

enum class RandomFloatType {
  Positive,
  Negative,
  Any,
};

template <typename T>
T generateRandomFloat(RandomFloatType fType = RandomFloatType::Any) {
  static_assert(
      std::is_floating_point<T>::value,
      "generateRandomFloats() can only be used with floating point types"
      );

  // Generate a random floating point number
  T val;
  do {
    val = generateFloatBits<T>();
  } while (isnan(val));

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
