#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class Triangle: public QFPTest::TestBase<T> {
protected:
  Triangle(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}
  virtual T getArea(const T a, const T b, const T c) = 0;

public:
  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
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
  }

private:
  using QFPTest::TestBase<T>::id;
};

#endif // TRIANGLE_HPP
