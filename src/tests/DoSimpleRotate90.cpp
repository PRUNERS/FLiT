#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
DoSR90Kernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  VectorCU<T> A(ti.vals, ti.length);

  VectorCU<T> expected(A.size());
  expected[0]=-A[1]; expected[1]=A[0]; expected[2]=A[2];

  auto done = A.rotateAboutZ_3d(M_PI/2);
  
  results[idx].s1 = done.L1Distance(expected);
  results[idx].s2 = done.LInfDistance(expected);
}

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
  virtual QFPTest::KernelFunction<T>* getKernel() { return DoSR90Kernel; }

protected:
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    QFPHelpers::Vector<T> A(ti.vals);
    QFPHelpers::Vector<T> expected = {-A[1], A[0], A[2]};
    QFPHelpers::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A = A.rotateAboutZ_3d(M_PI/2);
    QFPHelpers::info_stream << "Resulting vector: " << A << std::endl;
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, QFPHelpers::info_stream);
    return {std::pair<long double, long double>(A.L1Distance(expected),
						A.LInfDistance(expected)),
	0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(DoSimpleRotate90)
