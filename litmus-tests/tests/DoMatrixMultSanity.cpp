#include <flit.h>

#include <functional>
#include <typeinfo>

#include <cmath>
#include <cstdio>

template <typename T>
GLOBAL
void
DoMatrixMultSanityKernel(const flit::CuTestInput<T>* tiList, flit::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  auto b = flit::VectorCU<T>(ti.vals, ti.length);
  auto c = flit::MatrixCU<T>::Identity(ti.length) * b;
  results[idx].s1 = c.L1Distance(b);
  results[idx].s2 = c.LInfDistance(b);
}

template <typename T>
class DoMatrixMultSanity: public flit::TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 16; }

  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.highestDim = getInputsPerRun();
    ti.min = -6;
    ti.max = 6;
    ti.vals = std::move(
        flit::Vector<T>::getRandomVector(getInputsPerRun()).getData());
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return DoMatrixMultSanityKernel; }
  virtual
  flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
    auto dim = ti.vals.size();
    flit::Vector<T> b(ti.vals);
    auto c = flit::Matrix<T>::Identity(dim) * b;
    bool eq = (c == b);
    flit::info_stream << id << ": Product is: " << c << std::endl;
    flit::info_stream << id << ": A * b == b? " << eq << std::endl;
    return {std::pair<long double, long double>(c.L1Distance(b), c.LInfDistance(b)), 0};
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoMatrixMultSanity)
