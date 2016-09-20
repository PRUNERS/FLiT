#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"


template <typename T>
DEVICE
T getCArea(const T a,
          const T b,
          const T c){
    T s = (a + b + c ) / 2;
    return CUHelpers::csqrt((T)s * (s-a) * (s-b) * (s-c));
}

template <typename T>
T getArea(const T a,
          const T b,
          const T c){
    T s = (a + b + c ) / 2;
    return sqrt(s * (s-a) * (s-b) * (s-c));
}

template <typename T>
GLOBAL
void
TrianglePHKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
  using namespace CUHelpers;
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  T maxval = tiList[idx].vals[0];
  T a = maxval;
  T b = maxval;
  T c = maxval * csqrt((T)2.0);
  const T delta = maxval / (T)ti.iters;
  const T checkVal = (T)0.5 * b * a;

  double score = 0.0;

  for(T pos = 1; pos <= a; pos += delta){
    auto crit = getCArea(a,b,c);
    b = csqrt(cpow(pos, (T)2.0) +
	      cpow(maxval, (T)2.0));
    c = csqrt(cpow(a - pos, (T)2.0) +
	      cpow(maxval, (T)2.0));
    score += abs(crit - checkVal);
  }
  results[idx].s1 = score;
  results[idx].s2 = 0.0;
}

template <typename T>
class TrianglePHeron: public QFPTest::TestBase<T> {
public:
  TrianglePHeron(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.iters = 200;
    ti.vals = { 6.0 };
    return ti;
  }

protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return TrianglePHKern; }
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

REGISTER_TYPE(TrianglePHeron)

