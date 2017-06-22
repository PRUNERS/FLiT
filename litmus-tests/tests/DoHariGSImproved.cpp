#include "flit.h"

#include <cmath>
#include <typeinfo>

template <typename T>
GLOBAL
void
DoHGSITestKernel(const flit::CuTestInput<T>* tiList, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* vals = tiList[idx].vals;
  flit::VectorCU<T> a(vals, 3);
  flit::VectorCU<T> b(vals + 3, 3);
  flit::VectorCU<T> c(vals + 6, 3);

  auto r1 = a.getUnitVector();
  auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
  auto r3 = (c - r1 * (c ^ r1));
  r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();

  T o12 = r1 ^ r2;
  T o13 = r1 ^ r3;
  T o23 = r2 ^ r3;

  double score = std::abs(o12) + std::abs(o13) + std::abs(o23);

  results[idx] = score;
}

template <typename T>
class DoHariGSImproved: public flit::TestBase<T> {
public:
  DoHariGSImproved(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 9; }
  virtual flit::TestInput<T> getDefaultInput();

protected:
  virtual flit::KernelFunction<T>* getKernel() { return DoHGSITestKernel; }
  virtual long double run_impl(const flit::TestInput<T>& ti) {
    long double score = 0.0;

    //matrix = {a, b, c};
    flit::Vector<T> a = {ti.vals[0], ti.vals[1], ti.vals[2]};
    flit::Vector<T> b = {ti.vals[3], ti.vals[4], ti.vals[5]};
    flit::Vector<T> c = {ti.vals[6], ti.vals[7], ti.vals[8]};

    auto r1 = a.getUnitVector();
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    auto r3 = (c - r1 * (c ^ r1));
    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
    T o12 = r1 ^ r2;
    T o13 = r1 ^ r3;
    T o23 = r2 ^ r3;
    if((score = std::abs(o12) + std::abs(o13) + std::abs(o23)) != 0){
      flit::info_stream << id << ": in: " << id << std::endl;
      flit::info_stream << id << ": applied gram-schmidt to:" << std::endl;
      flit::info_stream << id << ":   a:  " << a << std::endl;
      flit::info_stream << id << ":   b:  " << b << std::endl;
      flit::info_stream << id << ":   c:  " << c << std::endl;
      flit::info_stream << id << ": resulting vectors were:" << std::endl;
      flit::info_stream << id << ":   r1: " << r1 << std::endl;
      flit::info_stream << id << ":   r2: " << r2 << std::endl;
      flit::info_stream << id << ":   r3: " << r3 << std::endl;
      flit::info_stream << id << ": w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
    }
    return score;
  }

protected:
  using flit::TestBase<T>::id;
};

namespace {
  template <typename T> T getSmallValue();
  template<> inline float getSmallValue() { return pow(10, -4); }
  template<> inline double getSmallValue() { return pow(10, -8); }
#ifndef __CUDA__
  template<> inline long double getSmallValue() { return pow(10, -10); }
#endif
} // end of unnamed namespace

template <typename T>
flit::TestInput<T> DoHariGSImproved<T>::getDefaultInput() {
  T e = getSmallValue<T>();

  flit::TestInput<T> ti;

  // Just one test
  ti.vals = {
    1, e, e,  // vec a
    1, e, 0,  // vec b
    1, 0, e,  // vec c
  };

  return ti;
}

REGISTER_TYPE(DoHariGSImproved)
