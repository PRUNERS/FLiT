// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "QFPHelpers.hpp"

#include <iostream>
#include <mutex>

namespace QFPHelpers {

std::vector<uint_fast32_t>
getShuffleSeq(uint_fast32_t size){
  std::vector<uint_fast32_t> retVal(size);
  iota(retVal.begin(), retVal.end(), 0);
  shuffle(retVal.begin(), retVal.end(), std::mt19937(RAND_SEED));
  return retVal;
}

std::vector<float>
setRandSequence(size_t size, int32_t seed){
  std::vector<float> ret(size);
  std::mt19937 gen;
  gen.seed(seed);
  std::uniform_real_distribution<float> dist(-6.0, 6.0);
  for(auto& i: ret) i = dist(gen);
  return ret;
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

const std::vector<float> float_rands = setRandSequence(RAND_VECT_SIZE);
const std::vector<uint_fast32_t> shuffled_16  = getShuffleSeq(16);

std::vector<float>
getRandSeq(){return float_rands;}

thread_local InfoStream info_stream;
std::mutex ostreamMutex;

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i){
  if(i == 0) os << 0;
  else{
    uint64_t hi = i >> 64;
    uint64_t lo = (uint64_t)i;
    ostreamMutex.lock();
    auto bflags = os.flags();
    os.flags(std::ios::hex & ~std::ios::showbase);
    os << "0x" << hi << lo;
    os.flags( bflags );
    ostreamMutex.unlock();
  }
  return os;
}

}
 
