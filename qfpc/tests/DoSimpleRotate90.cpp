#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class DoSimpleRotate90: public QFPTest::TestBase<T> {
public:
  DoSimpleRotate90(std::string id):QFPTest::TestBase<T>(std::move(id)) {}

  // TODO: Use these methods instead of canned test data in run_impl()
  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.iters = 200;
    ti.highestDim = 16;
    ti.ulp_inc = 1;
    ti.vals = { 1.0 }; // dummy value for now
    return ti;
  }

protected:

  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Q_UNUSED(ti);

    QFPHelpers::Vector<T> A = {1, 1, 1};
    QFPHelpers::Vector<T> expected = {-1, 1, 1};
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

REGISTER_TYPE(DoSimpleRotate90)
