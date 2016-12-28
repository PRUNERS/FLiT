#include "Kahan.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <iomanip>
#include <iostream>
#include <string>

template <typename T>
class KahanSum : public QFPTest::TestBase<T> {
public:
  KahanSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; } //10000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Kahan<T> kahan;
    T naive = 0.0;
    for (auto val : ti.vals) {
      kahan.add(val);
      naive += val;
      QFPHelpers::info_stream
      //std::cout
        //<< std::setiosflags(std::ios::scientific)
        << std::setw(7)
        << std::setprecision(7)
        << id << ": + " << val
        << " = " << kahan.sum() << " or " << naive
        << std::endl;
    }
    QFPHelpers::info_stream << id << ": naive sum = " << naive << std::endl;
    QFPHelpers::info_stream << id << ": kahan sum = " << kahan.sum() << std::endl;
    QFPHelpers::info_stream
      << id << ": sum = " << kahan.sum() << " or " << naive << std::endl;
    return {kahan.sum(), naive};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

namespace {
template<typename T> std::vector<T> getToRepeat();

//template<> std::vector<float> getToRepeat() { return { 1.0, 1.0e8, -1.0e8 }; }
template<> std::vector<float> getToRepeat() { return { 1.0e4, 3.14159, 2.71828 }; }
template<> std::vector<double> getToRepeat() { return { 1.0, 1.0e100, 1.0, -1.0e100 }; }
template<> std::vector<long double> getToRepeat() { return { 1.0, 1.0e200, 1.0, -1.0e200 }; }
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
