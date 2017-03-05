#include "Kahan.h"
#include "Shewchuk.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#define PI  3.14159265358979323846264338L
#define EXP 2.71828182845904523536028747L

template <typename T>
class KahanSum : public QFPTest::TestBase<T> {
public:
  KahanSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 10000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Kahan<T> kahan;
    Shewchuk<T> chuk;
    T naive = 0.0;
    for (auto val : ti.vals) {
      chuk.add(val);
      kahan.add(val);
      naive += val;
    }
    QFPHelpers::info_stream << id << ": pi           = " << static_cast<T>(PI) << std::endl;
    QFPHelpers::info_stream << id << ": exp(1)       = " << static_cast<T>(EXP) << std::endl;
    QFPHelpers::info_stream << id << ": naive sum    = " << naive << std::endl;
    QFPHelpers::info_stream << id << ": kahan sum    = " << kahan.sum() << std::endl;
    QFPHelpers::info_stream << id << ": shewchuk sum = " << kahan.sum() << std::endl;
    QFPHelpers::info_stream << id << ": Epsilon      = " << std::numeric_limits<T>::epsilon() << std::endl;
    return {std::pair<long double, long double>(kahan.sum(), naive), 0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

namespace {
  template<typename T> std::vector<T> getToRepeat();
//template<> std::vector<float> getToRepeat() { return { 1.0, 1.0e8, -1.0e8 }; }
  template<> std::vector<float> getToRepeat() { return { 1.0e4, PI, EXP, -1.0e4 }; }
  template<> std::vector<double> getToRepeat() { return { 1.0e11, PI, EXP, -1.0e11 }; }
#ifndef __CUDA__
  template<> std::vector<long double> getToRepeat() { return { 1.0e14, PI, EXP, -1.0e14 }; }
#endif
}

template <typename T>
QFPTest::TestInput<T> KahanSum<T>::getDefaultInput() {
  QFPTest::TestInput<T> ti;
  auto dim = getInputsPerRun();
  ti.highestDim = dim;
  ti.vals = std::vector<T>(dim);
  auto toRepeat = getToRepeat<T>();
  for (decltype(dim) i = 0, j = 0;
       i < dim;
       i++, j = (j+1) % toRepeat.size()) {
    ti.vals[i] = toRepeat[j];
  }
  return ti;
}

REGISTER_TYPE(KahanSum)
