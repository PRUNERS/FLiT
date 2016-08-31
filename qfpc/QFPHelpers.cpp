// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "QFPHelpers.hpp"

#include <iostream>
#include <mutex>

namespace QFPHelpers {

const std::vector<float> float_rands = setRandSequence<float>(256);
const std::vector<double> double_rands = setRandSequence<double>(256);
const std::vector<long double>  long_double_rands = setRandSequence<long double>(256);

template <>
std::vector<float>
getRandSeq<float>(){return float_rands;}

template <>
std::vector<double>
getRandSeq<double>(){return double_rands;}

template <>
std::vector<long double>
getRandSeq<long double>(){return long_double_rands;}
  

thread_local InfoStream info_stream;
std::mutex ostreamMutex;

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i){
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
  ostreamMutex.lock();
  auto bflags = os.flags();
  os.flags(std::ios::hex | std::ios::showbase);
  os << hi << lo;
  os.flags( bflags );
  ostreamMutex.unlock();
  return os;
}

}
 
