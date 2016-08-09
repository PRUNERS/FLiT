#include "groundtruth.h"
#include "test.h"

#include <random>
#include <tuple>
#include <iostream>
#include <cmath>

template <typename ToType, typename FromType>
ToType my_reinterpret(FromType val) {
  return *reinterpret_cast<ToType*>(&val);
}


template <typename T>
std::vector<std::tuple<T,T,T>> testDistribution(unsigned int numTrials) {
  using elementType = std::tuple<T,T,T>;
  std::vector<elementType> mismatchedInputs;

  std::random_device seed_generator;
  auto seed = seed_generator();
  std::cout << "mt seed:     " << seed << std::endl;
  std::mt19937 generator32(seed);
  std::mt19937_64 generator64(seed);

  for (unsigned int i = 0; i < numTrials; i++) {
    T a, b, c;
    if (sizeof(T) == 4) {
      a = my_reinterpret<T>(generator32());
      b = my_reinterpret<T>(generator32());
      c = my_reinterpret<T>(generator32());
    } else if (sizeof(T) == 8) {
      a = my_reinterpret<T>(generator64());
      b = my_reinterpret<T>(generator64());
      c = my_reinterpret<T>(generator64());
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

void runDistributionFloat() {
  const unsigned int count = 1000000;//99999999;  // number of experiments

  auto mismatches = testDistribution<float>(count);
  for (auto& input : mismatches) {
    printf("\n");
    printf("DistributionTest: Diverging inputs:\n");
    printf("DistributionTest:   0x%08x  %e\n", my_reinterpret<unsigned>(std::get<0>(input)), std::get<0>(input));
    printf("DistributionTest:   0x%08x  %e\n", my_reinterpret<unsigned>(std::get<1>(input)), std::get<1>(input));
    printf("DistributionTest:   0x%08x  %e\n", my_reinterpret<unsigned>(std::get<2>(input)), std::get<2>(input));
  }
  
  float a = my_reinterpret<float, unsigned>(0x6b8b4567);
  float b = my_reinterpret<float, unsigned>(0x65ba0c1e);
  float c = my_reinterpret<float, unsigned>(0x49e753d2);
  auto truth = distributionTruth(a, b, c);
  auto testval = distributionTest(a, b, c);
  
  if (truth != testval) {
    printf("\n");
    printf("truth      = 0x%08x\n", my_reinterpret<unsigned>(truth));
    printf("testval    = 0x%08x\n", my_reinterpret<unsigned>(testval));
    printf("difference = 0x%08x\n", my_reinterpret<unsigned>(truth - testval));
  }
}

void runDistributionDouble() {
  const unsigned int count = 1000000;//99999999;  // number of experiments

  auto mismatches = testDistribution<double>(count);
  for (auto& input : mismatches) {
    printf("\n");
    printf("DistributionTest: Diverging inputs:\n");
    printf("DistributionTest:   0x%016lx  %e\n", my_reinterpret<unsigned long>(std::get<0>(input)), std::get<0>(input));
    printf("DistributionTest:   0x%016lx  %e\n", my_reinterpret<unsigned long>(std::get<1>(input)), std::get<1>(input));
    printf("DistributionTest:   0x%016lx  %e\n", my_reinterpret<unsigned long>(std::get<2>(input)), std::get<2>(input));
  }
  
  auto a = my_reinterpret<double, unsigned long>(0x6b8b4567);
  auto b = my_reinterpret<double, unsigned long>(0x65ba0c1e);
  auto c = my_reinterpret<double, unsigned long>(0x49e753d2);
  auto truth = distributionTruth(a, b, c);
  auto testval = distributionTest(a, b, c);
  
  if (truth != testval) {
    printf("\n");
    printf("truth      = 0x%016lx\n", my_reinterpret<unsigned long>(truth));
    printf("testval    = 0x%016lx\n", my_reinterpret<unsigned long>(testval));
    printf("difference = 0x%016lx\n", my_reinterpret<unsigned long>(truth - testval));
  }
}

int main(void) {
  //std::cout << "float min:   " << std::numeric_limits<float>::min() << std::endl;
  //std::cout << "float max:   " << std::numeric_limits<float>::max() << std::endl;
  //std::cout << "double min:  " << std::numeric_limits<double>::min() << std::endl;
  //std::cout << "double max:  " << std::numeric_limits<double>::max() << std::endl;

  //runDistributionFloat();
  runDistributionDouble();

  return 0;
}

