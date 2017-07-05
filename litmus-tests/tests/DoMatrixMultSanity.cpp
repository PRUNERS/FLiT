#include <flit.h>

#include <functional>
#include <typeinfo>

#include <cmath>
#include <cstdio>

template <typename T>
GLOBAL
void
DoMatrixMultSanityKernel(const flit::CuTestInput<T>* tiList, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  auto b = flit::VectorCU<T>(ti.vals, ti.length);
  auto c = flit::MatrixCU<T>::Identity(ti.length) * b;
  results[idx] = c.L1Distance(b);
}

template <typename T>
class DoMatrixMultSanity: public flit::TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 16; }

  virtual flit::TestInput<T> getDefaultInput() override {
    flit::TestInput<T> ti;
    ti.highestDim = getInputsPerRun();
    ti.min = -6;
    ti.max = 6;
    ti.vals = std::move(
        flit::Vector<T>::getRandomVector(getInputsPerRun()).getData());
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return DoMatrixMultSanityKernel; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) override {
    auto dim = ti.vals.size();
    flit::Vector<T> b(ti.vals);
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
