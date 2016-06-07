// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include <unordered_map>

#include "QFPHelpers.h"

namespace QFPHelpers {
  
static 
void
printOnce(std::string s, void* addr){
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

std::default_random_engine gen;

}
