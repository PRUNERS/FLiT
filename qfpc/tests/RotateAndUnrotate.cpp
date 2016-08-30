#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class RotateAndUnrotate: public QFPTest::TestBase<T> {
public:
  RotateAndUnrotate(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
    T min = ti.min;
    T max = ti.max;
    auto theta = M_PI;
    auto A = QFPHelpers::Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    QFPHelpers::info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << std::endl;
    A.rotateAboutZ_3d(theta);
    QFPHelpers::info_stream << "Rotated is: " << A << std::endl;
    A.rotateAboutZ_3d(-theta);
    QFPHelpers::info_stream << "Unrotated is: " << A << std::endl;
    bool equal = A == orig;
    QFPHelpers::info_stream << "Are they equal? " << equal << std::endl;
    auto dist = A.L1Distance(orig);
    if(!equal){
      QFPHelpers::info_stream << "error in L1 distance is: " << dist << std::endl;
      QFPHelpers::info_stream << "difference between: " << (A - orig) << std::endl;
    }
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, QFPHelpers::info_stream);
    return {{{id, typeid(T).name()}, {dist, A.LInfDistance(orig)}}};
  }

private:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(RotateAndUnrotate)
