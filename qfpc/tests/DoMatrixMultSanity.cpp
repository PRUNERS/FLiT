#include <cmath>
#include <functional>
#include <typeinfo>

#include <stdio.h>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"


template <typename T>
GLOBAL
void
DoMatrixMultSanityKernel(const QFPTest::CuTestInput<T> ti, QFPTest::CudaResultElement* results){
  auto dim = ti.vals.size();
  auto b = CUHelpers::VectorCU<T>(ti.vals);
  auto c = CUHelpers::MatrixCU<T>::Identity(dim) * b;
  results->s1 = c.L1Distance(b);
  results->s2 = c.LInfDistance(b);
}


template <typename T>
class DoMatrixMultSanity: public QFPTest::TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 16; }

  virtual QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.highestDim = getInputsPerRun();
    ti.min = -6;
    ti.max = 6;
    ti.vals = std::move(
        QFPHelpers::Vector<T>::getRandomVector(getInputsPerRun()).getData());
    return ti;
  }

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() { return DoMatrixMultSanityKernel; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto dim = ti.vals.size();
    QFPHelpers::Vector<T> b(ti.vals);
    auto c = QFPHelpers::Matrix<T>::Identity(dim) * b;
    bool eq = (c == b);
    QFPHelpers::info_stream << id << ": Product is: " << c << std::endl;
    QFPHelpers::info_stream << id << ": A * b == b? " << eq << std::endl;
    return {c.L1Distance(b), c.LInfDistance(b)};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(DoMatrixMultSanity)
