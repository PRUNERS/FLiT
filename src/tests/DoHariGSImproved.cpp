#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

using namespace CUHelpers;

template <typename T>
GLOBAL
void
DoHGSITestKernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* vals = tiList[idx].vals;
  VectorCU<T> a(vals, 3);
  VectorCU<T> b(vals + 3, 3);
  VectorCU<T> c(vals + 6, 3);

  auto r1 = a.getUnitVector();
  auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
  auto r3 = (c - r1 * (c ^ r1));
  r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();

  T o12 = r1 ^ r2;
  T o13 = r1 ^ r3;
  T o23 = r2 ^ r3;

  double score = abs(o12) + abs(o13) + abs(o23);

  results[idx].s1 = score;
  results[idx].s2 = 0;
}

template <typename T>
class DoHariGSImproved: public QFPTest::TestBase<T> {
public:
  DoHariGSImproved(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 9; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() { return DoHGSITestKernel; }
  virtual QFPTest::ResultType::mapped_type
  run_impl(const QFPTest::TestInput<T>& ti) {
    long double score = 0.0;

    //matrix = {a, b, c};
    QFPHelpers::Vector<T> a = {ti.vals[0], ti.vals[1], ti.vals[2]};
    QFPHelpers::Vector<T> b = {ti.vals[3], ti.vals[4], ti.vals[5]};
    QFPHelpers::Vector<T> c = {ti.vals[6], ti.vals[7], ti.vals[8]};

    auto r1 = a.getUnitVector();
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    auto r3 = (c - r1 * (c ^ r1));
    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
    T o12 = r1 ^ r2;
    T o13 = r1 ^ r3;
    T o23 = r2 ^ r3;
    if((score = std::abs(o12) + std::abs(o13) + std::abs(o23)) != 0){
      QFPHelpers::info_stream << id << ": in: " << id << std::endl;
      QFPHelpers::info_stream << id << ": applied gram-schmidt to:" << std::endl;
      QFPHelpers::info_stream << id << ":   a:  " << a << std::endl;
      QFPHelpers::info_stream << id << ":   b:  " << b << std::endl;
      QFPHelpers::info_stream << id << ":   c:  " << c << std::endl;
      QFPHelpers::info_stream << id << ": resulting vectors were:" << std::endl;
      QFPHelpers::info_stream << id << ":   r1: " << r1 << std::endl;
      QFPHelpers::info_stream << id << ":   r2: " << r2 << std::endl;
      QFPHelpers::info_stream << id << ":   r3: " << r3 << std::endl;
      QFPHelpers::info_stream << id << ": w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
    }
    return {score, 0.0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

namespace {
  template <typename T> T getSmallValue();
  template<> inline float getSmallValue() { return pow(10, -4); }
  template<> inline double getSmallValue() { return pow(10, -8); }
#ifndef __CUDA__
  template<> inline long double getSmallValue() { return pow(10, -10); }
#endif
}

template <typename T>
QFPTest::TestInput<T> DoHariGSImproved<T>::getDefaultInput() {
  T e = getSmallValue<T>();

  QFPTest::TestInput<T> ti;

  // Just one test
  ti.vals = {
    1, e, e,  // vec a
    1, e, 0,  // vec b
    1, 0, e,  // vec c
  };

  return ti;
}

REGISTER_TYPE(DoHariGSImproved)
