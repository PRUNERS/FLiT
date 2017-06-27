#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
DEVICE
T getCArea(const T a,
          const T b,
          const T c){
    T s = (a + b + c ) / 2;
    return flit::csqrt((T)s * (s-a) * (s-b) * (s-c));
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
TrianglePHKern(const flit::CuTestInput<T>* tiList, double* results) {
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  T maxval = tiList[idx].vals[0];
  T a = maxval;
  T b = maxval;
  T c = maxval * flit::csqrt((T)2.0);
  const T delta = maxval / (T)ti.iters;
  const T checkVal = (T)0.5 * b * a;

  double score = 0.0;

  for(T pos = 0; pos <= a; pos += delta){
    b = flit::csqrt(flit::cpow(pos, (T)2.0) +
	      flit::cpow(maxval, (T)2.0));
    c = flit::csqrt(flit::cpow(a - pos, (T)2.0) +
	      flit::cpow(maxval, (T)2.0));
    auto crit = getCArea(a,b,c);
    score += std::abs(crit - checkVal);
  }
  results[idx] = score;
}

template <typename T>
class TrianglePHeron: public flit::TestBase<T> {
public:
  TrianglePHeron(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.iters = 200;
    ti.vals = { 6.0 };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() {return TrianglePHKern; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) {
    T maxval = ti.vals[0];
    // start as a right triangle
    T a = maxval;
    T b = maxval;
    T c = maxval * std::sqrt(2);
    const T delta = maxval / (T)ti.iters;

    // 1/2 b*h = A
    // all perturbations will have the same base and height (plus some FP noise)
    const T checkVal = 0.5 * b * a;

    long double score = 0;

    for(T pos = 0; pos <= a; pos += delta){
      b = std::sqrt(std::pow(pos, 2) +
                    std::pow(maxval, 2));
      c = std::sqrt(std::pow(a - pos, 2) +
                    std::pow(maxval, 2));
      auto crit = getArea(a,b,c);
      score += std::abs(crit - checkVal);
    }
    return score;
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(TrianglePHeron)

