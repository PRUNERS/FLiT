#ifndef DO_SIMPLE_ROTATE_90_HPP
#define DO_SIMPLE_ROTATE_90_HPP

#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class DoSimpleRotate90: public QFPTest::TestBase<T> {
public:
  DoSimpleRotate90(std::string id):QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.vals = { 1, 1, 1 };
    return ti;
  }

protected:

  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    QFPHelpers::Vector<T> A(ti.vals);
    QFPHelpers::Vector<T> expected = {-A[1], A[0], A[2]};
    QFPHelpers::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A.rotateAboutZ_3d(M_PI/2);
    QFPHelpers::info_stream << "Resulting vector: " << A << std::endl;
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, QFPHelpers::info_stream);
    return {A.L1Distance(expected), A.LInfDistance(expected)};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

#endif // DO_SIMPLE_ROTATE_90_HPP
