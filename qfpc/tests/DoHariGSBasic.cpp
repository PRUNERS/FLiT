#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class DoHariGSBasic: public QFPTest::TestBase {
public:
  DoHariGSBasic(std::string id) : QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
    Q_UNUSED(ti);
    using namespace QFPHelpers;

    //auto& crit = getWatchData<T>();
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
    QFPHelpers::Vector<T> a = {1, e, e};
    QFPHelpers::Vector<T> b = {1, e, 0};
    QFPHelpers::Vector<T> c = {1, 0, e};
    auto r1 = a.getUnitVector();
    //crit = r1[0];
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    //crit =r2[0];
    auto r3 = (c - r1 * (c ^ r1) -
               r2 * (c ^ r2)).getUnitVector();
    //crit = r3[0];
    T o12 = r1 ^ r2;
    //    crit = o12;
    T o13 = r1 ^ r3;
    //crit = o13;
    T o23 = r2 ^ r3;
    //crit = 023;
    if((score = std::abs(o12) + std::abs(o13) + std::abs(o23)) != 0){
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
      QFPHelpers::info_stream << "score (bits): " <<
        QFPHelpers::FPHelpers::as_int(score) << std::endl;
      QFPHelpers::info_stream << "score (dec) :" << score << std::endl;
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
  }
};

REGISTER_TYPE(DoHariGSBasic)

