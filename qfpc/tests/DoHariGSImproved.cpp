#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

#include "cudaTests.hpp"

using namespace CUHelpers;

template <typename T>
GLOBAL
void
DoHGSITestKernel(const QFPTest::testInput ti, cudaResultElement* results){
  T e;
  sizeof(T) == 4 ? e = powf(10, -4) : e = pow(10, -8);

  VectorCU<T> a(3);
  VectorCU<T> b(3);
  VectorCU<T> c(3);
  a[0]=1;a[1]=e;a[2]=e;
  b[0]=1;b[1]=e;b[2]=0;
  c[0]=1;c[1]=0;c[2]=e;

  auto r1 = a.getUnitVector();
  auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
  auto r3 = (c - r1 * (c ^ r1));
  r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();

  T o12 = r1 ^ r2;
  T o13 = r1 ^ r3;
  T o23 = r2 ^ r3;

  printf("o12:%lf o13:%lf o23:%lf\n",(double)o12, (double)o13, (double)o23);

  double score = abs(o12) + abs(o13) + abs(o23);

  results[0].s1 = score;
  results[0].s2 = 0;
}

template <typename T>
class DoHariGSImproved: public QFPTest::TestBase {
public:
  DoHariGSImproved(std::string id) : QFPTest::TestBase(id) {}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
    Q_UNUSED(ti);

#ifdef __CUDA__
    return DoCudaTest(ti, id, DoHGSITestKernel<T>, typeid(T).name(),
		      1);
#else
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    QFPHelpers::Vector<T> a = {1, e, e};
    QFPHelpers::Vector<T> b = {1, e, 0};
    QFPHelpers::Vector<T> c = {1, 0, e};

    auto r1 = a.getUnitVector();
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    auto r3 = (c - r1 * (c ^ r1));
    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
    T o12 = r1 ^ r2;
    T o13 = r1 ^ r3;
    T o23 = r2 ^ r3;
    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
      QFPHelpers::info_stream << "in: " << id << std::endl;
      QFPHelpers::info_stream << "applied gram-schmidt to:" << std::endl;
      QFPHelpers::info_stream << "a: " << a << std::endl;
      QFPHelpers::info_stream << "b: " << b << std::endl;
      QFPHelpers::info_stream << "c: " << c << std::endl;
      QFPHelpers::info_stream << "resulting vectors were: " << std::endl;
      QFPHelpers::info_stream << "r1: " << r1 << std::endl;
      QFPHelpers::info_stream << "r2: " << r2 << std::endl;
      QFPHelpers::info_stream << "r3: " << r3 << std::endl;
      QFPHelpers::info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
    }
    return {{
      {id, typeid(T).name()}, {score, 0.0}
    }};
#endif
  }
};

REGISTER_TYPE(DoHariGSImproved)
