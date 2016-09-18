#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
DoSkewSCPRKernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  const T* valData = tiList[idx].vals.data();
  auto A = VectorCU<T>(valData, 3).getUnitVector();
  auto B = VectorCU<T>(valData + 3, 3).getUnitVector();
  auto cross = A.cross(B);
  auto sine = cross.L2Norm();
  auto cos = A ^ B;
  auto sscpm = MatrixCU<T>::SkewSymCrossProdM(cross);
  auto rMatrix = MatrixCU<T>::Identity(3) +
    sscpm + (sscpm * sscpm) * ((1 - cos)/(sine * sine));
  auto result = rMatrix * A;
  results[idx].s1 = result.L1Distance(B);
  results[idx].s1 = result.LInfDistance(B);
}

template <typename T>
class DoSkewSymCPRotationTest: public QFPTest::TestBase<T> {
public:
  DoSkewSymCPRotationTest(std::string id)
    : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 6; }
  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.min = -6;
    ti.max = 6;
    auto n = getInputsPerRun();
    ti.highestDim = n;
    ti.vals = QFPHelpers::Vector<T>::getRandomVector(n).getData();
    return ti;
  }

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() { return DoSkewSCPRKernel;}

  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    //    auto& crit = getWatchData<T>();
    QFPHelpers::info_stream << "entered " << id << std::endl;
    long double L1Score = 0.0;
    long double LIScore = 0.0;
    QFPHelpers::Vector<T> A = { ti.vals[0], ti.vals[1], ti.vals[2] };
    QFPHelpers::Vector<T> B = { ti.vals[3], ti.vals[4], ti.vals[5] };
    A = A.getUnitVector();
    B = B.getUnitVector();
    QFPHelpers::info_stream << "A (unit) is: " << std::endl << A << std::endl;
    QFPHelpers::info_stream << "B (unit): " << std::endl  << B << std::endl;
    auto cross = A.cross(B); //cross product
    QFPHelpers::info_stream << "cross: " << std::endl << cross << std::endl;
    auto sine = cross.L2Norm();
    QFPHelpers::info_stream << "sine: " << std::endl << sine << std::endl;
    //    crit = A ^ B; //dot product
    auto cos = A ^ B;
    //    QFPHelpers::info_stream << "cosine: " << std::endl << crit << std::endl;
    QFPHelpers::info_stream << "cosine: " << std::endl << cos << std::endl;
    auto sscpm = QFPHelpers::Matrix<T>::SkewSymCrossProdM(cross);
    QFPHelpers::info_stream << "sscpm: " << std::endl << sscpm << std::endl;
    auto rMatrix = QFPHelpers::Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - cos)/(sine * sine));
    // auto rMatrix = QFPHelpers::Matrix<T>::Identity(3) +
    //   sscpm + (sscpm * sscpm) * ((1 - crit)/(sine * sine));
    auto result = rMatrix * A;
    QFPHelpers::info_stream << "rotator: " << std::endl << rMatrix << std::endl;
    if(!(result == B)){
      L1Score = result.L1Distance(B);
      LIScore = result.LInfDistance(B);
      QFPHelpers::info_stream << "Skew symmetric cross product rotation failed with ";
      QFPHelpers::info_stream << "L1Distance " << L1Score << std::endl;
      QFPHelpers::info_stream << "starting vectors: " << std::endl;
      QFPHelpers::info_stream << A << std::endl;
      QFPHelpers::info_stream << "...and..." << std::endl;
      QFPHelpers::info_stream << B << std::endl;
      QFPHelpers::info_stream << "ended up with: " << std::endl;
      QFPHelpers::info_stream << "L1Distance: " << L1Score << std::endl;
      QFPHelpers::info_stream << "LIDistance: " << LIScore << std::endl;
    }
    return {L1Score, LIScore};
  }

private:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(DoSkewSymCPRotationTest)
