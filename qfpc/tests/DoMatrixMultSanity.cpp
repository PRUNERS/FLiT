#include <cmath>
#include <functional>
#include <typeinfo>

#include <stdio.h>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"
#include "cudaTests.hpp"

template <typename T>
GLOBAL
void
DoMatrixMultSanityKernel(const QFPTest::testInput ti, cudaResultElement* results){
  auto dim = ti.highestDim;
  auto b = CUHelpers::VectorCU<T>::getRandomVector(dim);
  auto c = CUHelpers::MatrixCU<T>::Identity(dim) * b;
  results[0].s1 = c.L1Distance(b);
  results[0].s2 = c.LInfDistance(b);
}

template <typename T>
class DoMatrixMultSanity: public QFPTest::TestBase {
public:
  DoMatrixMultSanity(std::string id) : QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
#if defined(__CUDA__)
    return DoCudaTest(ti, id, DoMatrixMultSanityKernel<T>,
		      typeid(T).name(), 1);
#else
    auto dim = ti.highestDim;
    QFPHelpers::Vector<T> b = QFPHelpers::Vector<T>::getRandomVector(dim);
    auto c = QFPHelpers::Matrix<T>::Identity(dim) * b;
    QFPHelpers::info_stream << "Product is: " << c << std::endl;
    bool eq = c == b;
    QFPHelpers::info_stream << "A * b == b? " << eq << std::endl;
    return {{
      {id, typeid(T).name()}, {c.L1Distance(b), c.LInfDistance(b)}
    }};
#endif
  }
};

REGISTER_TYPE(DoMatrixMultSanity)
