#ifndef DO_MATRIX_MULT_SANITY_HPP
#define DO_MATRIX_MULT_SANITY_HPP

#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>


template <typename T>
class DoMatrixMultSanity: public QFPTest::TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 16; }

  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.highestDim = getInputsPerRun();
    ti.min = -6;
    ti.max = 6;
    ti.vals = QFPHelpers::Vector<T>::getRandomVector(
        ti.highestDim, ti.min, ti.max
        ).getData();
    return ti;
  }

protected:
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto dim = getInputsPerRun();
    QFPHelpers::Vector<T> b = {ti.vals[0], ti.vals[1], ti.vals[2]};
    auto c = QFPHelpers::Matrix<T>::Identity(dim) * b;
    bool eq = (c == b);
    QFPHelpers::info_stream << id << ": Product is: " << c << std::endl;
    QFPHelpers::info_stream << id << ": A * b == b? " << eq << std::endl;
    return {c.L1Distance(b), c.LInfDistance(b)};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

#endif // DO_MATRIX_MULT_SANITY_HPP
