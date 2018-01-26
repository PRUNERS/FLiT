#include <flit.h>

#include <typeinfo>

#include <cmath>

namespace {
  const int iters = 200;
} // end of unnamed namespace

template <typename T>
GLOBAL
void
RFCKern(const T* tiList, size_t n, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* ti = tiList + (idx*n);
  auto A = flit::VectorCU<T>(ti, n);
  auto orig = A;
  T theta = 2 * M_PI / iters;
  for(int r = 0; r < iters; ++r){
    A = A.rotateAboutZ_3d(theta);
  }
  results[idx] = A.L1Distance(orig);
}

template <typename T>
class RotateFullCircle: public flit::TestBase<T> {
public:
  RotateFullCircle(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    auto n = getInputsPerRun();
    return flit::Vector<T>::getRandomVector(n).getData();
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override {return RFCKern; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    flit::Vector<T> A = flit::Vector<T>(ti);
    auto orig = A;
    T theta = 2 * M_PI / iters;
    flit::info_stream << "Rotate full circle in " << iters
                      << " increments, A is: " << A << std::endl;
    for(int r = 0; r < iters; ++r){
      A.rotateAboutZ_3d(theta);
      flit::info_stream << r << " rotations, vect = " << A << std::endl;
    }
    flit::info_stream << "Rotated is: " << A << std::endl;
    bool equal = A == orig;
    flit::info_stream << "Does rotated vect == starting vect? " << equal
                      << std::endl;
    if(!equal){
      flit::info_stream << "The (vector) difference is: " << (A - orig)
                        << std::endl;
    }
    flit::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, flit::info_stream);
    return A.L1Distance(orig);
  }

private:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(RotateFullCircle)
