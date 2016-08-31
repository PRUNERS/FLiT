#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class RotateAndUnrotate: public QFPTest::TestBase<T> {
public:
  RotateAndUnrotate(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  // TODO: Use these methods instead of canned test data in run_impl()
  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.min = -6.0;
    ti.max = 6.0;
    ti.iters = 200;
    ti.highestDim = 16;
    ti.ulp_inc = 1;
    ti.vals = { 1.0 }; // dummy value for now
    return ti;
  }

protected:
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    T min = ti.min;
    T max = ti.max;
    auto theta = M_PI;
    auto A = QFPHelpers::Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    QFPHelpers::info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << std::endl;
    A.rotateAboutZ_3d(theta);
    QFPHelpers::info_stream << "Rotated is: " << A << std::endl;
    A.rotateAboutZ_3d(-theta);
    QFPHelpers::info_stream << "Unrotated is: " << A << std::endl;
    bool equal = A == orig;
    QFPHelpers::info_stream << "Are they equal? " << equal << std::endl;
    auto dist = A.L1Distance(orig);
    if(!equal){
      QFPHelpers::info_stream << "error in L1 distance is: " << dist << std::endl;
      QFPHelpers::info_stream << "difference between: " << (A - orig) << std::endl;
    }
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, QFPHelpers::info_stream);
    return {dist, A.LInfDistance(orig)};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(RotateAndUnrotate)
