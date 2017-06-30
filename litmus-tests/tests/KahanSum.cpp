#include "Kahan.h"
#include "Shewchuk.h"

#include <flit.h>

#include <iomanip>
#include <iostream>
#include <string>

#define PI  3.14159265358979323846264338L
#define EXP 2.71828182845904523536028747L

template <typename T>
class KahanSum : public flit::TestBase<T> {
public:
  KahanSum(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 10000; }
  virtual flit::TestInput<T> getDefaultInput() override;

protected:
  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) override {
    Kahan<T> kahan;
    Shewchuk<T> chuk;
    T naive = 0.0;
    for (auto val : ti.vals) {
      chuk.add(val);
      kahan.add(val);
      naive += val;
    }
    T kahan_sum = kahan.sum();
    T shewchuk_sum = chuk.sum();
    flit::info_stream << id << ": pi           = " << static_cast<T>(PI) << std::endl;
    flit::info_stream << id << ": exp(1)       = " << static_cast<T>(EXP) << std::endl;
    flit::info_stream << id << ": naive sum    = " << naive << std::endl;
    flit::info_stream << id << ": kahan sum    = " << kahan_sum << std::endl;
    flit::info_stream << id << ": shewchuk sum = " << shewchuk_sum << std::endl;
    flit::info_stream << id << ": Epsilon      = " << std::numeric_limits<T>::epsilon() << std::endl;
    return kahan_sum;
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
