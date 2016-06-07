// This is where the tests live for QFP.
// All tests need to derive from TestBase,
// and call registerTest.

#include "testBase.h"
#include "QFPHelpers.h"

namespace QFPTest {

using namespace QFPHelpers;
  
class DoSkewSymCPRotationTest: public TestBase {
public:
  DoSkewSymCPRotationTest():TestBase("DoSkewSymCPRotationTest"){}
  resultType floatTest(const testInput& ti) override {
    return operator()<float>(ti);
  }
  resultType doubleTest(const testInput& ti) override {
    return operator()<double>(ti);
  }
  resultType longTest(const testInput& ti) override {
    return operator()<long double>(ti);
  }
  template <typename T>
  resultType operator()(const testInput& ti) {
    auto& min = ti.min;
    auto& max = ti.max;
    auto& crit = getWatchData<T>();
    info_stream << "entered " << __func__ << std::endl; 
    long double L1Score = 0.0;
    long double LIScore = 0.0;
    auto A = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "A (unit) is: " << std::endl << A << std::endl;
    auto B = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "B (unit): " << std::endl  << B << std::endl;
    auto cross = A.cross(B); //cross product
    info_stream << "cross: " << std::endl << cross << std::endl;
    auto sine = cross.L2Norm();
    info_stream << "sine: " << std::endl << sine << std::endl;
    crit = A ^ B; //dot product
    info_stream << "cosine: " << std::endl << crit << std::endl;
    auto sscpm = Matrix<T>::SkewSymCrossProdM(cross);
    info_stream << "sscpm: " << std::endl << sscpm << std::endl;
    auto rMatrix = Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - crit)/(sine * sine));
    auto result = rMatrix * A;
    info_stream << "rotator: " << std::endl << rMatrix << std::endl;
    if(!(result == B)){
      L1Score = result.L1Distance(B);
      LIScore = result.LInfDistance(B);
      info_stream << "Skew symmetric cross product rotation failed with ";
      info_stream << "L1Distance " << L1Score << std::endl;
      info_stream << "starting vectors: " << std::endl;
      info_stream << A << std::endl;
      info_stream << "...and..." << std::endl;
      info_stream << B << std::endl;
      info_stream << "ended up with: " << std::endl;
      info_stream << "L1Distance: " << L1Score << std::endl;
      info_stream << "LIDistance: " << LIScore << std::endl;
    }
    return {__func__, {L1Score, LIScore}};
  }
};

REGISTER_TYPE(DoSkewSymCPRotationTest)
  
} //namespace QFPTest
