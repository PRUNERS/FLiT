#include "Kahan.h"
#include "Shewchuk.h"

#include "TestBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#define PI  3.14159265358979323846264338L
#define EXP 2.71828182845904523536028747L

template <typename T>
class KahanSum : public flit::TestBase<T> {
public:
  KahanSum(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 10000; }
  virtual flit::TestInput<T> getDefaultInput();

protected:
  virtual flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
    Kahan<T> kahan;
    Shewchuk<T> chuk;
    T naive = 0.0;
    for (auto val : ti.vals) {
      chuk.add(val);
      kahan.add(val);
      naive += val;
    }
    flit::info_stream << id << ": pi           = " << static_cast<T>(PI) << std::endl;
    flit::info_stream << id << ": exp(1)       = " << static_cast<T>(EXP) << std::endl;
    flit::info_stream << id << ": naive sum    = " << naive << std::endl;
    flit::info_stream << id << ": kahan sum    = " << kahan.sum() << std::endl;
    flit::info_stream << id << ": shewchuk sum = " << kahan.sum() << std::endl;
    flit::info_stream << id << ": Epsilon      = " << std::numeric_limits<T>::epsilon() << std::endl;
    return {std::pair<long double, long double>(kahan.sum(), naive), 0};
  }

protected:
  using flit::TestBase<T>::id;
};

namespace {
  template<typename T> std::vector<T> getToRepeat();
//template<> std::vector<float> getToRepeat() { return { 1.0, 1.0e8, -1.0e8 }; }
  template<> std::vector<float> getToRepeat() { return { 1.0e4, PI, EXP, -1.0e4 }; }
  template<> std::vector<double> getToRepeat() { return { 1.0e11, PI, EXP, -1.0e11 }; }
#ifndef __CUDA__
  template<> std::vector<long double> getToRepeat() { return { 1.0e14, PI, EXP, -1.0e14 }; }
#endif
} // end of unnamed namespace

template <typename T>
flit::TestInput<T> KahanSum<T>::getDefaultInput() {
  flit::TestInput<T> ti;
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
