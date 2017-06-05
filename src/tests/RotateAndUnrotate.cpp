#include <cmath>
#include <typeinfo>

#include "TestBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"


template <typename T>
GLOBAL
void
RaUKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
  using namespace CUHelpers;
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto theta = M_PI;
  auto ti = tiList[idx];
  auto A = VectorCU<T>(ti.vals, ti.length);
  auto orig = A;
  A = A.rotateAboutZ_3d(theta);
  A = A.rotateAboutZ_3d(-theta);
  results[idx].s1 = A.L1Distance(orig);
  results[idx].s2 = A.LInfDistance(orig);
}

template <typename T>
class RotateAndUnrotate: public QFPTest::TestBase<T> {
public:
  RotateAndUnrotate(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    ti.vals = QFPHelpers::Vector<T>::getRandomVector(3).getData();
    return ti;
  }

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() { return RaUKern; }
  virtual
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
    return {std::pair<long double, long double>(dist, A.LInfDistance(orig)), 0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(RotateAndUnrotate)
