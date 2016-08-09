#include "groundtruth.h"

namespace {

  template<typename T>
  T distributionTruth_impl(T a, T b, T c) {
    return (a * c) + (b * c);
  }

}

float distributionTruth(float a, float b, float c) {
  return distributionTruth_impl(a, b, c);
}
double distributionTruth(double a, double b, double c) {
  return distributionTruth_impl(a, b, c);
}
long double distributionTruth(long double a, long double b, long double c) {
  return distributionTruth_impl(a, b, c);
}
