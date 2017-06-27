// These are compensating algorithms that use FMA to calculate
// an EFT (error-free transformation)
// see http://perso.ens-lyon.fr/nicolas.louvet/LaLo07b.pdf

#include <flit.h>

#include <tuple>

#include <cmath>

//this is a dummy -- needs to be implemented
//the body of a different algo in CUDA is left
//for instructional purposes
// template <typename T>
// GLOBAL
// void
// addNameHere(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   auto ti = tiList[idx];
//   T maxval = tiList[idx].vals[0];
//   T a = maxval;
//   T b = maxval;
//   T c = maxval * flit::csqrt((T)2.0);
//   const T delta = maxval / (T)ti.iters;
//   const T checkVal = (T)0.5 * b * a;

//   double score = 0.0;

//   for(T pos = 0; pos <= a; pos += delta){
//     b = flit::csqrt(flit::cpow(pos, (T)2.0) +
// 	      flit::cpow(maxval, (T)2.0));
//     c = flit::csqrt(flit::cpow(a - pos, (T)2.0) +
// 	      flit::cpow(maxval, (T)2.0));
//     auto crit = getCArea(a,b,c);
//     score += std::abs(crit - checkVal);
//   }
//   results[idx] = score;
// }

//these are the helpers for the langois compensating algos
//will be executed in their own right as well as supporting
//other tests in this file

namespace{
template <typename T>
void
TwoSum(T a, T b, T& x, T&y){
  x = a * b;
  T z = x - a;
  y = (a - (x - z)) + (b - z);
}

template <typename T>
void
TwoProd(T a, T b, T& x, T& y){
  x = a * b;
  y = std::fma(a, b, -x);
}

template <typename T>
void
ThreeFMA(T a, T b, T c, T& x, T& y, T& z){
  T u1, u2, a1, B1, B2;
  x = std::fma(a, b, c);
  TwoProd(a, b, u1, u2);
  TwoSum(b, u2, a1, z);
  TwoSum(u1, a1, B1, B2);
  y = (B1 - x) + B2;
}
} // end of unnamed namespace

//algorithm 11
template <typename T>
class langDotFMA: public flit::TestBase<T> {
public:
  langDotFMA(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 0; }
  virtual flit::TestInput<T> getDefaultInput() { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = flit::getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    s[0] = x[0] * y[0]; 
    for(stype i = 1; i < size; ++i){
      s[i] = std::fma(x[i], y[i], s[i-1]);
    }
    return s[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langDotFMA)

//algorithm 12
template <typename T>
class langCompDotFMA: public flit::TestBase<T> {
public:
  langCompDotFMA(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 0; }
  virtual flit::TestInput<T> getDefaultInput() { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = flit::getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    std::vector<T> c(size);
    T a, B;
    TwoProd(x[0], y[0], s[0], c[0]);
    for(stype i = 1; i < size; ++i){
      ThreeFMA(x[i], y[i], s[i-1], s[i], a, B);
      c[i] = c[i-1] + (a + B);
    }
    return s[size-1] + c[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langCompDotFMA)

//algorithm 13
template <typename T>
class langCompDot: public flit::TestBase<T> {
public:
  langCompDot(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 0; }
  virtual flit::TestInput<T> getDefaultInput() { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = flit::getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    std::vector<T> c(size);
    T pi, si, p;
    TwoProd(x[0], y[0], s[0], c[0]);
    for(stype i = 1; i < size; ++i){
      TwoProd(x[i], y[i], p, pi);
      TwoSum(p, s[i-1], s[i], si);
      c[i] = c[i-1] + (pi + si);
    }
    return s[size-1] + c[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langCompDot)
