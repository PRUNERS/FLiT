// the header for FLiT helper functions.  These classes, such as matrix and
// vector, utilize the TestBase watch data items for monitoring by differential
// debugging.

#ifndef FLIT_HELPERS_HPP
#define FLIT_HELPERS_HPP

#include "InfoStream.h"
#include "CUHelpers.h"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <ostream>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include <cfloat>

#ifndef FLIT_UNUSED
#define FLIT_UNUSED(x) (void)x
#endif

namespace flit {

const int RAND_SEED = 1;
const int RAND_VECT_SIZE = 256;

extern thread_local InfoStream info_stream;

// this section provides a pregenerated random
// sequence that can be used by tests, including
// CUDA

template <typename T>
const std::vector<T>
setRandSeq(size_t size, int32_t seed = RAND_SEED){
  // there may be a bug with float uniform_real_dist
  // it is giving very different results than double or long double
  std::vector<T> ret(size);
  std::mt19937 gen;
  gen.seed(seed);
  std::uniform_real_distribution<double> dist(-6.0, 6.0);
  for(auto& i: ret) i = T(dist(gen));
  return ret;
}
  
const std::vector<uint_fast32_t>
getShuffleSeq(uint_fast32_t);

extern const std::vector<float> float_rands;
extern const std::vector<double> double_rands;
extern const std::vector<long double> long_rands;

extern const std::vector<uint_fast32_t> shuffled_16;

template <typename T>
std::vector<T> const &
getRandSeq();

std::ostream& operator<<(std::ostream&, const unsigned __int128);
unsigned __int128 stouint128(const std::string &str);

template <typename F, typename I>
HOST_DEVICE
F as_float_impl(I val) {
  static_assert(sizeof(F) == sizeof(I), "cannot convert types of different sizes");
  union {
    I i;
    F f;
  } u = { val };
  return u.f;
}

HOST_DEVICE
inline float
as_float(uint32_t val) {
  return as_float_impl<float, uint32_t>(val);
}

HOST_DEVICE
inline double
as_float(uint64_t val) {
  return as_float_impl<double, uint64_t>(val);
}

inline long double
as_float(unsigned __int128 val) {
  return as_float_impl<long double, __int128>(val);
}

template <typename F, typename I>
HOST_DEVICE
I as_int_impl(F val) {
  static_assert(sizeof(F) == sizeof(I), "cannot convert types of different sizes");
  union {
    F f;
    I i;
  } u = { val };
  return u.i;
}

HOST_DEVICE
inline uint32_t
as_int(float val) {
  return as_int_impl<float, uint32_t>(val);
}

HOST_DEVICE
inline uint64_t
as_int(double val) {
  return as_int_impl<double, uint64_t>(val);
}

inline unsigned __int128
as_int(long double val) {
  const unsigned __int128 zero = 0;
  const auto temp = as_int_impl<long double, __int128>(val);
  return temp & (~zero >> 48);
}

} // end of namespace flit

#endif // FLIT_HELPERS_HPP
 
