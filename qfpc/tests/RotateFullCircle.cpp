#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"
#include "cudaTests.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
RFCKern(const QFPTest::testInput ti, cudaResultElement* results){
  auto n = ti.iters;
  auto A = VectorCU<T>::getRandomVector(3);
  auto orig = A;
  T theta = 2 * M_PI / n;
  for(decltype(n) r = 0; r < n; ++r){
    A = A.rotateAboutZ_3d(theta);
  }
  results[0].s1 = A.L1Distance(orig);
  results[0].s2 = A.LInfDistance(orig);
}

template <typename T>
class RotateFullCircle: public QFPTest::TestBase<T> {
public:
  RotateFullCircle(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    ti.iters = 200;
    auto n = getInputsPerRun();
    ti.highestDim = n;
    ti.vals = QFPHelpers::Vector<T>::getRandomVector(n).getData();
    return ti;
  }

protected:

  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto n = ti.iters;
    QFPHelpers::Vector<T> A = QFPHelpers::Vector<T>(ti.vals);
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
