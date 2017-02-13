#include "Shewchuk.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <iomanip>
#include <string>
#include <vector>

template <typename T>
class ShewchukSum : public QFPTest::TestBase<T> {
public:
  ShewchukSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}
  
  virtual size_t getInputsPerRun() { return 1000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Shewchuk<T> chuk;
    T naive = 0.0;
    for (auto val : ti.vals) {
      chuk.add(val);
      naive += val;
      //QFPHelpers::info_stream
      //  << std::setw(7)
      //  << std::setprecision(7)
      //  << id << ": + " << val
      //  << " = " << chuk.sum() << " or " << naive
      //  << std::endl;
			QFPHelpers::info_stream
				<< id << ":   partials now: (" << chuk.partials().size() << ") ";
      for (auto p : chuk.partials()) {
        QFPHelpers::info_stream << " " << p;
      }
      QFPHelpers::info_stream << std::endl;
    }
    T sum = chuk.sum();
    QFPHelpers::info_stream << id << ": naive sum    = " << naive << std::endl;
    QFPHelpers::info_stream << id << ": shewchuk sum = " << sum << std::endl;
    QFPHelpers::info_stream << id << ": shewchuk partials = " << chuk.partials().size() << std::endl;
    return {sum, chuk.sum2()};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

namespace {
template<typename T> std::vector<T> getToRepeat();

template<> std::vector<float> getToRepeat() { return { 1.0, 1.0e8, 1.0, -1.0e8 }; }
template<> std::vector<double> getToRepeat() { return { 1.0, 1.0e100, 1.0, -1.0e100 }; }
template<> std::vector<long double> getToRepeat() { return { 1.0, 1.0e200, 1.0, -1.0e200 }; }
}

template <typename T>
QFPTest::TestInput<T> ShewchukSum<T>::getDefaultInput() {
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

REGISTER_TYPE(ShewchukSum)
