#include "groundtruth.h"
#include "test.h"

#include <typeinfo>
#include <random>
#include <tuple>
#include <iostream>
#include <cmath>
#include <type_traits>

template <typename ToType, typename FromType>
ToType my_reinterpret(FromType val) {
  return *reinterpret_cast<ToType*>(&val);
}


// returns a bitlength equivalent unsigned type for floats
// and a bitlength equivalent floating type for integral types
template<typename T>
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
template<typename F>
get_corresponding_type_t<F>
reinterpret_convert(F val) {
  using ToType = get_corresponding_type_t<F>;
  return *reinterpret_cast<ToType*>(&val);
}

/** Convenience for reinterpret_convert().  Enforces the type to be integral */
template<typename I>
decltype(auto) reinterpret_int_as_float(I val) {
  static_assert(std::is_integral<I>::value
                  || std::is_same<I, __int128>::value
                  || std::is_same<I, unsigned __int128>::value,
                "Must pass an integral type to reinterpret as floating-point");
  return reinterpret_convert(val);
}

/** Convenience for reinterpret_convert().  Enforces the type to be floating-point */
template<typename F>
decltype(auto) reinterpret_float_as_int(F val) {
  static_assert(std::is_floating_point<F>::value,
                "Must pass a floating-point type to reinterpret as integral");
  return reinterpret_convert(val);
}

template <typename T>
std::vector<std::tuple<T,T,T>> testDistribution(unsigned int numTrials) {
  using elementType = std::tuple<T,T,T>;
  std::vector<elementType> mismatchedInputs;

  std::random_device seed_generator;
  auto seed = seed_generator();
  std::mt19937 generator32(seed);
  std::mt19937_64 generator64(seed);
  auto generator128 = [&generator64]() {
    unsigned __int128 val = generator64();
    val = val << 64;
    val += generator64();
    return val;
  };

  for (unsigned int i = 0; i < numTrials; i++) {
    T a, b, c;
    if (sizeof(T) == 4) {
      a = reinterpret_int_as_float(static_cast<uint32_t>(generator32()));
      b = reinterpret_int_as_float(static_cast<uint32_t>(generator32()));
      c = reinterpret_int_as_float(static_cast<uint32_t>(generator32()));
    } else if (sizeof(T) == 8) {
      a = reinterpret_int_as_float(static_cast<uint64_t>(generator64()));
      b = reinterpret_int_as_float(static_cast<uint64_t>(generator64()));
      c = reinterpret_int_as_float(static_cast<uint64_t>(generator64()));
    } else if (sizeof(T) == 16) {
      a = reinterpret_int_as_float(generator128());
      b = reinterpret_int_as_float(generator128());
      c = reinterpret_int_as_float(generator128());
    } else {
      throw std::runtime_error("Error");
    }
    a = std::abs(a);
    b = std::abs(b);
    c = std::abs(c);
    T truth = distributionTruth(a, b, c);
    T testval = distributionTest(a, b, c);
    
    if (truth != testval) {
      mismatchedInputs.emplace_back(a, b, c);
    }
  }

  return mismatchedInputs;
}

void runDistributionFloat(const unsigned int count) {
  auto mismatches = testDistribution<float>(count);
  auto printAsInt32 = [](float val) {
    auto intval = reinterpret_float_as_int(val);
    printf("DistributionTest:   0x%08x  %e\n", intval, val);
  };
  for (auto& input : mismatches) {
    printf("\n");
    printf("DistributionTest: Diverging inputs (float):\n");
    printAsInt32(std::get<0>(input));
    printAsInt32(std::get<1>(input));
    printAsInt32(std::get<2>(input));
  }
}

void runDistributionDouble(const unsigned int count) {
  auto mismatches = testDistribution<double>(count);
  auto printAsInt64 = [](double val) {
    auto intval = reinterpret_float_as_int(val);
    printf("DistributionTest:   0x%016lx  %e\n", intval, val);
  };
  for (auto& input : mismatches) {
    printf("\n");
    printf("DistributionTest: Diverging inputs (double):\n");
    printAsInt64(std::get<0>(input));
    printAsInt64(std::get<1>(input));
    printAsInt64(std::get<2>(input));
  }
}

void runDistributionLongDouble(const unsigned int count) {
  auto mismatches = testDistribution<long double>(count);
  auto printAsInt80 = [](long double val) {
    auto intval = reinterpret_float_as_int(val);
    printf("DistributionTest:   0x%04lx%016lx  %Le\n",
           static_cast<uint64_t>(intval >> 64),
           static_cast<uint64_t>(intval),
           val);
  };
  for (auto& input : mismatches) {
    printf("\n");
    printf("DistributionTest: Diverging inputs (long double):\n");
    printAsInt80(std::get<0>(input));
    printAsInt80(std::get<1>(input));
    printAsInt80(std::get<2>(input));
  }
}

int main(void) {
  runDistributionFloat(100);
  runDistributionDouble(100);
  runDistributionLongDouble(20);

  return 0;
}

