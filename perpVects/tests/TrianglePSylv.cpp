#include "Triangle.hpp"
#include "QFPHelpers.h"

#include <cmath>

template <typename T>
class TrianglePSylv: public Triangle<T> {
public:
  TrianglePSylv(std::string id):Triangle<T>(id){}

  T
  getArea(const T a,
          const T b,
          const T c){
    return (pow(2.0, -2)*sqrt((a+(b+c))*(a+(b-c))*(c+(a-b))*(c-(a-b))));
  }

};

REGISTER_TYPE(TrianglePSylv)
