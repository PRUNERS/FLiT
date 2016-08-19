// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "QFPHelpers.h"

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

InfoStream info_stream;
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

std::mutex ostreamMutex;

std::string
getSortName(sort_t val){
  switch(val){
  case lt:
    return "lt";
  case gt:
    return "gt";
  case bi:
    return "bi";
  case def:
    return "us";
  default:
    return "something bad happened, undefined sort type";
  }
}
}
