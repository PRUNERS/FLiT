//this is the base instantiation for tests

#include "testBase.hpp"

#include <stack>

// namespace  {
// double volatile baseD;
// bool dReg = false;
// float volatile baseF;
// bool fReg = false;
// long double volatile baseL;
// bool lReg = false;
// bool watching = false;
// std::stack<float> fStack;
// std::stack<double> dStack;
// std::stack<long double> lStack;
// }
namespace QFPTest{

  //output operator for ResultType
std::ostream&
operator<<(std::ostream& os, const ResultType& res){
  // std::string name = r.first;
  // std::string prec;
  // long double s1;
  // long double s2;
  // std::tie(prec, s1, s2) = r.second;
  for(auto r  : res){
    os << r.first.first << ":" << r.first.second << ","
       << r.second.first << "," << r.second.second << std::endl;
  }
  return os;
}
}
