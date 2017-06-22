#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
GLOBAL
void
RaUKern(const flit::CuTestInput<T>* tiList, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto theta = M_PI;
  auto ti = tiList[idx];
  auto A = flit::VectorCU<T>(ti.vals, ti.length);
  auto orig = A;
  A = A.rotateAboutZ_3d(theta);
  A = A.rotateAboutZ_3d(-theta);
  results[idx] = A.L1Distance(orig);
}

template <typename T>
class RotateAndUnrotate: public flit::TestBase<T> {
public:
  RotateAndUnrotate(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    ti.vals = flit::Vector<T>::getRandomVector(3).getData();
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return RaUKern; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto theta = M_PI;
    auto A = flit::Vector<T>(ti.vals);
    auto orig = A;
    flit::info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << std::endl;
    A.rotateAboutZ_3d(theta);
    flit::info_stream << "Rotated is: " << A << std::endl;
    A.rotateAboutZ_3d(-theta);
    flit::info_stream << "Unrotated is: " << A << std::endl;
    bool equal = A == orig;
    flit::info_stream << "Are they equal? " << equal << std::endl;
    auto dist = A.L1Distance(orig);
    if(!equal){
      flit::info_stream << "error in L1 distance is: " << dist << std::endl;
      flit::info_stream << "difference between: " << (A - orig) << std::endl;
    }
    flit::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, flit::info_stream);
    return dist;
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(RotateAndUnrotate)
