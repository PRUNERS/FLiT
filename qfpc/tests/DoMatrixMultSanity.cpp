#include "testBase.hpp"
#include "QFPHelpers.hpp"
//#include "DoMatrixMultSanity.cuh"

#include <cmath>
#include <typeinfo>

template <typename T>
QFPTest::resultType DoMatrixMultSanity_CUDA(const QFPTest::testInput&, std::string);

template <typename T>
class DoMatrixMultSanity: public QFPTest::TestBase {
public:
  DoMatrixMultSanity(std::string id) : QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
#ifdef __CUDA__
    return DoMatrixMultSanity_CUDA<T>(ti, id);
#else
    auto dim = ti.highestDim;
    // T min = ti.min;
    // T max = ti.max;
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
