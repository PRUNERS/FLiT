#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

template <typename T>
class DoSkewSymCPRotationTest: public QFPTest::TestBase<T> {
public:
  DoSkewSymCPRotationTest(std::string id)
    : QFPTest::TestBase<T>(std::move(id)) {}

  QFPTest::ResultType run(const QFPTest::TestInput& ti) {
    auto& min = ti.min;
    auto& max = ti.max;
    //    auto& crit = getWatchData<T>();
    QFPHelpers::info_stream << "entered " << id << std::endl;
    long double L1Score = 0.0;
    long double LIScore = 0.0;
    auto A = QFPHelpers::Vector<T>::getRandomVector(3, min, max).getUnitVector();
    QFPHelpers::info_stream << "A (unit) is: " << std::endl << A << std::endl;
    auto B = QFPHelpers::Vector<T>::getRandomVector(3, min, max).getUnitVector();
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
    return {{
      {id, typeid(T).name()},
            {L1Score, LIScore}
    }};
  }

private:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(DoSkewSymCPRotationTest)
