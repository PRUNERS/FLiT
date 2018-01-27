#include <flit.h>

#include <typeinfo>

#include <cmath>

namespace {
  int g_iters = 200;
}

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
TrianglePHKern(const T* tiList, size_t n, double* results) {
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto start = tiList + (idx*n);
  T maxval = start[0];
  T a = maxval;
  T b = maxval;
  T c = maxval * flit::csqrt((T)2.0);
  const T delta = maxval / T(g_iters);
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

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { 6.0 };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override {return TrianglePHKern; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T maxval = ti[0];
    // start as a right triangle
    T a = maxval;
    T b = maxval;
    T c = maxval * std::sqrt(2);
    const T delta = maxval / T(g_iters);

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

