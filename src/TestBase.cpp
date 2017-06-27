//this is the base instantiation for tests

#include "TestBase.h"

#include <stack>

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
       << r.second.first << "," << r.second.second << std::endl;
  }
  return os;
}
} // end of namespace flit
