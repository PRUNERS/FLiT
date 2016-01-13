#include <iostream>
#include <cstdlib>
#include <cmath>


int
main(int argc, char* argv[]){
  float p1 = atof(argv[1]);
  float c1 = atof(argv[2]);
  double p2 = atof(argv[1]);
  double c2 = atof(argv[2]);
  long double p3 = atof(argv[1]);
  long double c3 = atof(argv[2]);
  std::cout << "parsed two double values from input" << std::endl;
  std::cout.precision(200);
  float fa = p1/c1;
  double da = p2/c2;
  long double lda = p3/c3;
  int exp;
  std::cout << "float p / c is: " << fa << std::endl;
  // float fsig = frexp(fa, &exp);
  // std::cout << "fa exp: " << exp << ", mantissa: " << fsig << std::endl;
  std::cout << "double p / c is: " << da << std::endl;
  // double dsig = frexp(da, &exp);
  // std::cout << "da exp: " << exp << ", mantissa: " << dsig << std::endl;
  std::cout << "long double p / c is: " << lda << std::endl;
  // long double ldsig = frexp(lda, &exp);
  // std::cout << "lda exp: " << exp << ", mantissa: " << ldsig << std::endl;

}
