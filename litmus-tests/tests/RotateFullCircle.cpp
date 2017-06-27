#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
GLOBAL
void
RFCKern(const flit::CuTestInput<T>* tiList, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto ti = tiList[idx];
  auto n = ti.iters;
  auto A = flit::VectorCU<T>(ti.vals, ti.length);
  auto orig = A;
  T theta = 2 * M_PI / n;
  for(decltype(n) r = 0; r < n; ++r){
    A = A.rotateAboutZ_3d(theta);
  }
  results[idx] = A.L1Distance(orig);
}

template <typename T>
class RotateFullCircle: public flit::TestBase<T> {
public:
  RotateFullCircle(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    ti.iters = 200;
    auto n = getInputsPerRun();
    ti.highestDim = n;
    ti.vals = flit::Vector<T>::getRandomVector(n).getData();
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() {return RFCKern; }

  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) {
    auto n = ti.iters;
    flit::Vector<T> A = flit::Vector<T>(ti.vals);
    auto orig = A;
    T theta = 2 * M_PI / n;
    flit::info_stream << "Rotate full circle in " << n << " increments, A is: " << A << std::endl;
    for(decltype(n) r = 0; r < n; ++r){
      A.rotateAboutZ_3d(theta);
      flit::info_stream << r << " rotations, vect = " << A << std::endl;
    }
    flit::info_stream << "Rotated is: " << A << std::endl;
    bool equal = A == orig;
    flit::info_stream << "Does rotated vect == starting vect? " << equal << std::endl;
    if(!equal){
      flit::info_stream << "The (vector) difference is: " << (A - orig) << std::endl;
    }
    flit::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, flit::info_stream);
    return A.L1Distance(orig);
  }

private:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(RotateFullCircle)
