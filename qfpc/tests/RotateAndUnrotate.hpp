#ifndef ROTATE_AND_UNROTATE_HPP
#define ROTATE_AND_UNROTATE_HPP

#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class RotateAndUnrotate: public QFPTest::TestBase<T> {
public:
  RotateAndUnrotate(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    ti.vals = QFPHelpers::Vector<T>::getRandomVector(3, ti.min, ti.max).getData();
    return ti;
  }

protected:
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto theta = M_PI;
    auto A = QFPHelpers::Vector<T>(ti.vals);
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

#endif // ROTATE_AND_UNROTATE_HPP
