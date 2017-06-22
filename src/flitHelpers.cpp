// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "flitHelpers.hpp"

#include <iostream>
#include <mutex>

namespace flit {

const std::vector<uint_fast32_t>
getShuffleSeq(uint_fast32_t size){
  std::vector<uint_fast32_t> retVal(size);
  iota(retVal.begin(), retVal.end(), 0);
  shuffle(retVal.begin(), retVal.end(), std::mt19937(RAND_SEED));
  return retVal;
}

template <>
float
get_tiny1<float>(){
  return 1.175494351-38;
}

template <>
double
get_tiny1<double>(){
  return 2.2250738585072014e-308;
}

template <>
long double
get_tiny1<long double>(){
  return 3.362103143112093506262e-4931L;
}

template <>
float
get_tiny2<float>(){
  return 1.175494352-38;
}

template <>
double
get_tiny2<double>(){
  return 2.2250738585072015e-308;
}

template <>
long double
get_tiny2<long double>(){
  return 3.362103143112093506263e-4931L;
}

  //const std::vector<float> float_rands = setRandSequence<float>(RAND_VECT_SIZE);

template<>
const std::vector<float>&
getRandSeq<float>(){return float_rands;}

template<>
const std::vector<double>&
getRandSeq<double>(){return double_rands;}

template<>
const std::vector<long double>&
getRandSeq<long double>(){return long_rands;}

const std::vector<float> float_rands = setRandSeq<float>(RAND_VECT_SIZE);
const std::vector<double> double_rands = setRandSeq<double>(RAND_VECT_SIZE);
const std::vector<long double> long_rands = setRandSeq<long double>(RAND_VECT_SIZE);

  
thread_local InfoStream info_stream;
std::mutex ostreamMutex;

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i){
  if(i == 0) os << 0;
  else{
    std::ostringstream ost;
    uint64_t hi = i >> 64;
    uint64_t lo = (uint64_t)i;
    ostreamMutex.lock();
    auto bflags = os.flags();
    os.flags(std::ios::hex & ~std::ios::showbase);
    ost.flags(std::ios::hex & ~std::ios::showbase);
    ost << lo;    
    os << "0x" << hi;
    for(uint32_t x = 0; x < 16 - ost.str().length(); ++x){
      os << "0";
    }
    os << ost.str();
    os.flags( bflags );
    ostreamMutex.unlock();
  }
  return os;
}

} // end of namespace flit
 
