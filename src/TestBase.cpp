//this is the base instantiation for tests

#include "TestBase.hpp"

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
// } // end of unnamed namespace
namespace flit {

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
       << r.second.first.first << "," << r.second.first.second << r.second.second << std::endl;
  }
  return os;
}
} // end of namespace flit
