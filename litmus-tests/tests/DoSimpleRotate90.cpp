#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
GLOBAL
void
DoSR90Kernel(const flit::CuTestInput<T>* tiList, flit::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  flit::VectorCU<T> A(ti.vals, ti.length);

  flit::VectorCU<T> expected(A.size());
  expected[0]=-A[1]; expected[1]=A[0]; expected[2]=A[2];

  auto done = A.rotateAboutZ_3d(M_PI/2);
  
  results[idx].s1 = done.L1Distance(expected);
  results[idx].s2 = done.LInfDistance(expected);
}

template <typename T>
class DoSimpleRotate90: public flit::TestBase<T> {
public:
  DoSimpleRotate90(std::string id):flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = { 1, 1, 1 };
    return ti;
  }
  virtual flit::KernelFunction<T>* getKernel() { return DoSR90Kernel; }

protected:
  flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
    flit::Vector<T> A(ti.vals);
    flit::Vector<T> expected = {-A[1], A[0], A[2]};
    flit::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A = A.rotateAboutZ_3d(M_PI/2);
    flit::info_stream << "Resulting vector: " << A << std::endl;
    flit::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, flit::info_stream);
    return {std::pair<long double, long double>(A.L1Distance(expected),
						A.LInfDistance(expected)),
	0};
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoSimpleRotate90)
