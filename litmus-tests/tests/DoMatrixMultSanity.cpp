#include <flit.h>

#include <functional>
#include <typeinfo>

#include <cmath>
#include <cstdio>

template <typename T>
GLOBAL
void
DoMatrixMultSanityKernel(const T* const* tiList, size_t n, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* ti = tiList[idx*n];
  auto b = flit::VectorCU<T>(ti, n);
  auto c = flit::MatrixCU<T>::Identity(n) * b;
  results[idx] = c.L1Distance(b);
}

template <typename T>
class DoMatrixMultSanity: public flit::TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 16; }

  virtual std::vector<T> getDefaultInput() override {
    return flit::Vector<T>::getRandomVector(getInputsPerRun()).getData();
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override {
    return DoMatrixMultSanityKernel;
  }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto dim = ti.size();
    flit::Vector<T> b(ti);
    auto c = flit::Matrix<T>::Identity(dim) * b;
    bool eq = (c == b);
    flit::info_stream << id << ": Product is: " << c << std::endl;
    flit::info_stream << id << ": A * b == b? " << eq << std::endl;
    return c.L1Distance(b);
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoMatrixMultSanity)
