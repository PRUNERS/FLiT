#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"
#include "cudaTests.hpp"

using namespace CUHelpers;
using namespace std::placeholders;

template <typename T>
class Triangle: public QFPTest::TestBase {
protected:
  Triangle(std::string id) : QFPTest::TestBase(id) {}
  HOST_DEVICE
  virtual T getArea(const T a, const T b, const T c) = 0;

public:
  GLOBAL
  void
  TriangleKern(const QFPTest::testInput ti, cudaResultElement* results){
    T a = ti.max;
    T b = ti.max;
    T c = csqrt(cpow(a,2.0f) + cpow(b, 2.0f));
    const T delta = ti.max / (T)ti.iters;

    const T checkVal = 0.5 * b * a;
    
    long double score = 0;

    for(T pos = 1; pos <= a; pos += delta){
      auto crit = getArea(a,b,c);
      b = csqrt(cpow(pos, 2.0f) +
		cpow(ti.max, 2.0f));
      c = csqrt(cpow(a - pos, 2.0f) +
		cpow(ti.max, 2.0f));
      score += abs(crit - checkVal);
    }
    results[0].s1 = score;
    results[0].s2 = 0.0;
  }
  
  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
#ifdef __CUDA__
    auto kern = [this](const QFPTest::testInput ti, cudaResultElement* re){
      return this->TriangleKern<<<1,1>>>(ti, re);
    };
    
    return DoCudaTest(ti, id, kern, typeid(T).name(), 1);
#else
    T a = ti.max;
    T b = ti.max;
    T c = std::sqrt(std::pow(a,2) + std::pow(b, 2));
    const T delta = ti.max / (T)ti.iters;

    // auto& crit = getWatchData<T>();

    // 1/2 b*h = A
    const T checkVal = 0.5 * b * a;  //all perturbations will have the same base and height

    long double score = 0;

    for(T pos = 1; pos <= a; pos += delta){
      auto crit = getArea(a,b,c);
      // crit = getArea(a,b,c);
      b = std::sqrt(std::pow(pos, 2) +
                    std::pow(ti.max, 2));
      c = std::sqrt(std::pow(a - pos, 2) +
                    std::pow(ti.max, 2));
      score += std::abs(crit - checkVal);
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
#endif
  }
};

#endif // TRIANGLE_HPP
