#include <flit.h>

#include <typeinfo>

#include <cmath>

template <typename T>
GLOBAL
void
DoSkewSCPRKernel(const T* tiList, size_t n, double* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* vals = tiList + (idx*n);
  auto A = flit::VectorCU<T>(vals, 3).getUnitVector();
  auto B = flit::VectorCU<T>(vals + 3, 3).getUnitVector();
  auto cross = A.cross(B);
  auto sine = cross.L2Norm();
  auto cos = A ^ B;
  auto sscpm = flit::MatrixCU<T>::SkewSymCrossProdM(cross);
  auto rMatrix = flit::MatrixCU<T>::Identity(3) +
    sscpm + (sscpm * sscpm) * ((1 - cos)/(sine * sine));
  auto result = rMatrix * A;
  results[idx] = result.L1Distance(B);
}

template <typename T>
class DoSkewSymCPRotationTest: public flit::TestBase<T> {
public:
  DoSkewSymCPRotationTest(std::string id)
    : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 6; }
  virtual std::vector<T> getDefaultInput() override {
    auto n = getInputsPerRun();
    return flit::Vector<T>::getRandomVector(n).getData();
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return DoSkewSCPRKernel;}

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    flit::info_stream << "entered " << id << std::endl;
    long double L1Score = 0.0;
    flit::Vector<T> A = { ti[0], ti[1], ti[2] };
    flit::Vector<T> B = { ti[3], ti[4], ti[5] };
    A = A.getUnitVector();
    B = B.getUnitVector();
    flit::info_stream << "A (unit) is: " << std::endl << A << std::endl;
    flit::info_stream << "B (unit): " << std::endl  << B << std::endl;
    auto cross = A.cross(B); //cross product
    flit::info_stream << "cross: " << std::endl << cross << std::endl;
    auto sine = cross.L2Norm();
    flit::info_stream << "sine: " << std::endl << sine << std::endl;
    //    crit = A ^ B; //dot product
    auto cos = A ^ B;
    //    flit::info_stream << "cosine: " << std::endl << crit << std::endl;
    flit::info_stream << "cosine: " << std::endl << cos << std::endl;
    auto sscpm = flit::Matrix<T>::SkewSymCrossProdM(cross);
    flit::info_stream << "sscpm: " << std::endl << sscpm << std::endl;
    auto rMatrix = flit::Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - cos)/(sine * sine));
    // auto rMatrix = flit::Matrix<T>::Identity(3) +
    //   sscpm + (sscpm * sscpm) * ((1 - crit)/(sine * sine));
    auto result = rMatrix * A;
    flit::info_stream << "rotator: " << std::endl << rMatrix << std::endl;
    if(!(result == B)){
      L1Score = result.L1Distance(B);
      flit::info_stream << "Skew symmetric cross product rotation failed with ";
      flit::info_stream << "L1Distance " << L1Score << std::endl;
      flit::info_stream << "starting vectors: " << std::endl;
      flit::info_stream << A << std::endl;
      flit::info_stream << "...and..." << std::endl;
      flit::info_stream << B << std::endl;
      flit::info_stream << "ended up with: " << std::endl;
      flit::info_stream << "L1Distance: " << L1Score << std::endl;
    }
    return L1Score;
  }

private:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoSkewSymCPRotationTest)
