#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class DoHariGSImproved: public QFPTest::TestBase {
public:
  DoHariGSImproved(std::string id) : QFPTest::TestBase(id) {}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
    Q_UNUSED(ti);

#ifdef __CUDA__
    return {{{id, typeid(T).name()}, {0.0, 0.0}}};
#else
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
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
