// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "QFPHelpers.hpp"

#include <iostream>
#include <unordered_map>
#include <mutex>

namespace QFPHelpers {

void printOnce(std::string s, void* addr){
  return;
  static std::unordered_map<void*, std::string> seen;
  if(seen.count(addr) == 0){
    seen.insert({addr, s});
    std::cout << s << " at: " << addr << std::endl;
    asm("int $3");
  }
}

thread_local InfoStream info_stream;
std::mutex ostreamMutex;

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i) {
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
  os << hi << lo;
  return os;
}

std::ostream& operator<<(std::ostream& os, const unsigned __int128 &i){
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
  os << hi << lo;
  return os;
}

namespace FPHelpers{
  float
  swap_float_int(uint32_t val){
    return *reinterpret_cast<float*>(&val);
  }

  double
  swap_float_int(uint64_t val){
     return *reinterpret_cast<double*>(&val);
  }

  long double
  swap_float_int(unsigned __int128 val){
    return *reinterpret_cast<long double*>(&val);
  }

  uint32_t
  swap_float_int(float val){
    return *reinterpret_cast<uint32_t*>(&val);
  }

  uint64_t
  swap_float_int(double val){
    return *reinterpret_cast<uint64_t*>(&val);
  }

  unsigned __int128
  swap_float_int(long double val){
    return *reinterpret_cast<unsigned __int128*>(&val);
  }
}

}
