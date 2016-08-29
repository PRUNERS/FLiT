// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "QFPHelpers.hpp"

#include <iostream>
#include <mutex>

namespace QFPHelpers {

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
 
