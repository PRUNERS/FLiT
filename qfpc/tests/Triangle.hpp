#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

using namespace CUHelpers;
using namespace std::placeholders;

template <typename T, typename Klass>
GLOBAL
void
TriangleKernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  
  auto ti = tiList[idx];
  T maxval = ti.vals[0];
  // start as a right triangle
  T a = maxval;
  T b = maxval;
  T c = maxval * sqrt(2.0);
  const T delta = maxval / (T)ti.iters;

  // 1/2 b*h = A
  // all perturbations will have the same base and height (plus some FP noise)
  const T checkVal = 0.5 * b * a;

  long double score = 0;

  for(T pos = 1; pos <= a; pos += delta){
    auto crit = getArea(a,b,c);
    b = sqrt(pow(pos, 2) +
             pow(maxval, 2));
    c = sqrt(pow(a - pos, 2) +
             pow(maxval, 2));
    score += abs(crit - checkVal);
  }
  results[idx].s1 = score;
  results[idx].s2 = 0.0;
}

template <typename T>
class Triangle: public QFPTest::TestBase<T> {
protected:
  HOST_DEVICE Triangle() = default;
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
  HOST_DEVICE
  virtual T getArea(const T a, const T b, const T c) = 0;

  // TODO: figure out how to get this one to work.
  virtual QFPTest::KernelFunction<T>* getKernel() { return nullptr; } //TriangleKernel<T, decltype(*this)>; }

  virtual
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
