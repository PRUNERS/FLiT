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
class Triangle: public QFPTest::TestBase<T> {
protected:
  Triangle(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

public:
  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.iters = 200;
    ti.vals = { 6.0 };
    return ti;
  }

protected:
  virtual T getArea(const T a, const T b, const T c) = 0;

  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    T maxval = ti.vals[0];
    // start as a right triangle
    T a = maxval;
    T b = maxval;
    T c = maxval * std::sqrt(2);
    const T delta = maxval / (T)ti.iters;

    // auto& crit = getWatchData<T>();

    // 1/2 b*h = A
    // all perturbations will have the same base and height (plus some FP noise)
    const T checkVal = 0.5 * b * a;

    long double score = 0;

    for(T pos = 1; pos <= a; pos += delta){
      auto crit = getArea(a,b,c);
      // crit = getArea(a,b,c);
      b = std::sqrt(std::pow(pos, 2) +
                    std::pow(maxval, 2));
      c = std::sqrt(std::pow(a - pos, 2) +
                    std::pow(maxval, 2));
      score += std::abs(crit - checkVal);
    }
    return {score, 0.0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

#endif // TRIANGLE_HPP
