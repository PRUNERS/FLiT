#include <cmath>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"
#include "cudaTests.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
DoSR90Kernel(const QFPTest::testInput ti, cudaResultElement* results){
  Q_UNUSED(ti);
  VectorCU<T> A(3);
  A[0] = 1; A[1] = 1; A[2] = 1;

  VectorCU<T> expected = A;
  expected[0]=-1; expected[1]=1; expected[2]=1;

  auto done = A.rotateAboutZ_3d(M_PI/2);
  
  results[0].s1 = done.L1Distance(expected);
  results[0].s2 = done.LInfDistance(expected);
}




template <typename T>
class DoSimpleRotate90: public QFPTest::TestBase {
public:
  DoSimpleRotate90(std::string id):QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
    Q_UNUSED(ti);

#ifdef __CUDA__
    return DoCudaTest(ti, id, DoSR90Kernel<T>,
		      typeid(T).name(), 1);
#else
    QFPHelpers::Vector<T> A = {1, 1, 1};
    QFPHelpers::Vector<T> expected = {-1, 1, 1};
    QFPHelpers::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    auto rot = A.rotateAboutZ_3d(M_PI/2);
    QFPHelpers::info_stream << "Resulting vector: " << rot << std::endl;
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, QFPHelpers::info_stream);
    return {{
      {id, typeid(T).name()}, {rot.L1Distance(expected), rot.LInfDistance(expected)}
    }};
#endif
  }
};

REGISTER_TYPE(DoSimpleRotate90)
