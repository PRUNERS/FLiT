#include <iostream>
#include <cstdlib>


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
  std::cout << "float p / c is: " << p1 / c1 << std::endl;

  std::cout << "double p / c is: " << p2 / c2 << std::endl;

  std::cout << "long double p / c is: " << p3 / c3 << std::endl;

}
