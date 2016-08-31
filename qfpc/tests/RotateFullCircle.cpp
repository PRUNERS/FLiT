#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class RotateFullCircle: public QFPTest::TestBase<T> {
public:
  RotateFullCircle(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

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
    auto n = ti.iters;
    T min = ti.min;
    T max = ti.max;
    QFPHelpers::Vector<T> A = QFPHelpers::Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    T theta = 2 * M_PI / n;
    QFPHelpers::info_stream << "Rotate full circle in " << n << " increments, A is: " << A << std::endl;
    for(decltype(n) r = 0; r < n; ++r){
      A.rotateAboutZ_3d(theta);
      QFPHelpers::info_stream << r << " rotations, vect = " << A << std::endl;
    }
    QFPHelpers::info_stream << "Rotated is: " << A << std::endl;
    bool equal = A == orig;
    QFPHelpers::info_stream << "Does rotated vect == starting vect? " << equal << std::endl;
    if(!equal){
      QFPHelpers::info_stream << "The (vector) difference is: " << (A - orig) << std::endl;
    }
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, QFPHelpers::info_stream);
    return {A.L1Distance(orig), A.LInfDistance(orig)};
  }

private:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(RotateFullCircle)
