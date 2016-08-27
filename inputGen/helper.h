#ifndef HELPER_H
#define HELPER_H

#include <type_traits>
#include <random>
#include <stdexcept>

namespace {

auto generateSeed() {
  std::random_device seedGenerator;
  return seedGenerator();
}

}

void printTestVal(const char* funcName, float val);
void printTestVal(const char* funcName, double val);
void printTestVal(const char* funcName, long double val);

// returns a bitlength equivalent unsigned type for floats
// and a bitlength equivalent floating type for integral types
template <typename T>
struct get_corresponding_type {
  using type = typename std::conditional_t<
    std::is_floating_point<T>::value && sizeof(T) == 4, uint32_t,
    std::conditional_t<
      std::is_floating_point<T>::value && sizeof(T) == 8, uint64_t,
		std::conditional_t<
      std::is_floating_point<T>::value && sizeof(T) == 16, unsigned __int128,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(float), float,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(double), double,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(long double), long double,
    std::conditional_t<
      std::is_same<T, __int128>::value && sizeof(long double) == 16, long double,
    std::conditional_t<
      std::is_same<T, unsigned __int128>::value && sizeof(long double) == 16, long double,
      void
    >>>>>>>>;
};

template <typename T>
using get_corresponding_type_t = typename get_corresponding_type<T>::type;

/**
 * Reinterpret float to integral or integral to float
 *
 * The reinterpreted float value will be an unsigned integral the same size as the passed-in float
 *
 * The reinterpreted integral value will be a floating point value of the same size
 *
 * Example:
 *   auto floatVal = reinterpret_convert(0x1f3742ae);
 *   auto intVal   = reinterpret_convert(floatVal);
 *   printf("%e\n", floatVal);
 *   printf("0x%08x\n", intVal);
 * Output:
 *   3.880691e-20
 *   0x1f3742ae
 */
template <typename F>
get_corresponding_type_t<F>
reinterpret_convert(F val) {
  using ToType = get_corresponding_type_t<F>;
  return *reinterpret_cast<ToType*>(&val);
}

/** Convenience for reinterpret_convert().  Enforces the type to be integral */
template <typename I>
decltype(auto) reinterpret_int_as_float(I val) {
  static_assert(std::is_integral<I>::value
                  || std::is_same<I, __int128>::value
                  || std::is_same<I, unsigned __int128>::value,
                "Must pass an integral type to reinterpret as floating-point");
  return reinterpret_convert(val);
}

/** Convenience for reinterpret_convert().  Enforces the type to be floating-point */
template <typename F>
decltype(auto) reinterpret_float_as_int(F val) {
  static_assert(std::is_floating_point<F>::value,
                "Must pass a floating-point type to reinterpret as integral");
  return reinterpret_convert(val);
}

enum class RandomFloatType {
  Positive,
  Negative,
  Any,
};

template <typename T>
std::vector<T> generateRandomFloats(uint count, RandomFloatType fType = RandomFloatType::Any) {
  static_assert(
      std::is_floating_point<T>::value,
      "generateRandomFloats() can only be used with floating point types"
      );

  static auto seed = generateSeed();
  static std::mt19937 generator32(seed);
  static std::mt19937_64 generator64(seed);
  static auto generator128 = []() {
    unsigned __int128 val = generator64();
    val = val << 64;
    val += generator64();
    return val;
  };

  std::vector<T> returnValues;

  while (returnValues.size() < count) {
    T val;
    if (sizeof(T) == 4) {
      val = reinterpret_int_as_float(static_cast<uint32_t>(generator32()));
    } else if (sizeof(T) == 8) {
      val = reinterpret_int_as_float(static_cast<uint64_t>(generator64()));
    } else if (sizeof(T) == 16) {
      val = reinterpret_int_as_float(generator128());
    } else {
      throw std::runtime_error("Error");
    }

    // If val is not a number (NaN), then try again
    if (isnan(val)) {
      continue;
    }

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

    returnValues.push_back(val);
  }

  return returnValues;
}

#endif // HELPER_H
