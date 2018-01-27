#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
GLOBAL
void
DoSR90Kernel(const T* tiList, size_t n, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* ti = tiList + (idx*n);
  flit::VectorCU<T> A(ti, n);

  flit::VectorCU<T> expected(A.size());
  expected[0]=-A[1]; expected[1]=A[0]; expected[2]=A[2];

  auto done = A.rotateAboutZ_3d(M_PI/2);
  
  results[idx] = done.L1Distance(expected);
}

template <typename T>
class DoSimpleRotate90: public flit::TestBase<T> {
public:
  DoSimpleRotate90(std::string id):flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return { 1, 1, 1 };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return DoSR90Kernel; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    flit::Vector<T> A(ti);
    flit::Vector<T> expected = {-A[1], A[0], A[2]};
    flit::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A = A.rotateAboutZ_3d(M_PI/2);
    flit::info_stream << "Resulting vector: " << A << std::endl;
    flit::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, flit::info_stream);
    return A.L1Distance(expected);
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoSimpleRotate90)
