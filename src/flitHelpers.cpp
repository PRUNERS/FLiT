// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "flitHelpers.h"

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

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i){
  std::ostringstream ost;
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
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
  return os;
}

unsigned __int128 stouint128(const std::string &str) {
  uint64_t hi, lo;
  // TODO: make this more efficient (maybe).
  std::string copy;
  if (str[0] == '0' && str[1] == 'x') {
    copy = std::string(str.begin() + 2, str.end());
  } else {
    copy = str;
  }

  // Convert each section of 8-bytes (16 characters)
  if (copy.size() > 32) {
    throw std::invalid_argument("Too many digits to convert with stouint128");
  }
  if (copy.size() <= 16) {
    hi = 0;
    lo = std::stoull(copy, nullptr, 16); 
  } else {
    auto mid = copy.end() - 16;
    hi = std::stoull(std::string(copy.begin(), mid), nullptr, 16);
    lo = std::stoull(std::string(mid, copy.end()), nullptr, 16);
  }

  // Combine the two 64-bit values.
  unsigned __int128 val;
  val = hi;
  val = val << 64;
  val += lo;
  return val;
}

} // end of namespace flit
 
