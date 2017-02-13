//this is the base instantiation for tests

#include "testBase.hpp"
#include "../gdb/inst/inst"

#include <stack>

namespace  {
double volatile baseD;
bool dReg = false;
float volatile baseF;
bool fReg = false;
long double volatile baseL;
bool lReg = false;
bool watching = false;
std::stack<float> fStack;
std::stack<double> dStack;
std::stack<long double> lStack;
}
namespace QFPTest{

using namespace QFP;

template<>
void
pushWatchData<float>(){
  float* tmp = const_cast<float*>(&baseF);
  fStack.push(*tmp);
}
template<>
void
popWatchData<float>(){
  baseF = fStack.top();
  fStack.pop();
}

template<>
void
pushWatchData<double>(){
  double* tmp = const_cast<double*>(&baseD);
  dStack.push(*tmp);
}
template<>
void
popWatchData<double>(){
  baseD = dStack.top();
  dStack.pop();
}

template<>
void
pushWatchData<long double>(){
  long double* tmp = const_cast<long double*>(&baseL);
  lStack.push(*tmp);
}
template<>
void
popWatchData<long double>(){
  baseL = lStack.top();
  lStack.pop();
}

template<>
volatile float& getWatchData<float>(){
  if(!fReg && watching){
    checkpoint(baseF, "testBase_float_watch");
    fReg = true;
  }
  return baseF;
}

template<>
volatile double& getWatchData<double>(){
  if(!dReg && watching){
    checkpoint(baseD, "testBase_double_watch");
    dReg = true;
  }
  return baseD;
}

template<>
volatile long double& getWatchData<long double>(){
  if(!lReg && watching){
    checkpoint(baseL, "testBase_long_double_watch");
    lReg = true;
  }
  return baseL;
}

void
setWatching(bool watch){
  watching = watch;
}

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
