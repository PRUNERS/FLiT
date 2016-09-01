#ifndef TRIANGLE_P_SYLV_HPP
#define TRIANGLE_P_SYLV_HPP

#include "Triangle.hpp"
#include "QFPHelpers.hpp"

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

#endif // TRIANGLE_P_SYLV_HPP
