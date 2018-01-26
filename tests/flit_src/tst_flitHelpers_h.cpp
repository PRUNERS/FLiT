#include "test_harness.h"

#include "flitHelpers.h"

#include "TestBase.h"   // for operator<<(flit::TestResult ...)

#include <algorithm>
#include <array>
#include <sstream>
#include <vector>

#include <cstdio>
#include <cstring>  // for memcpy()

namespace {

unsigned __int128 combine_to_128(uint64_t left_half, uint64_t right_half) {
  unsigned __int128 val = left_half;
  val = val << 64;
  val += right_half;
  return val;
}

} // end of unnamed namespace

template <typename T>
void tst_setRandSeq() {
  size_t n = 10;
  int32_t seed = 5024;
  auto expected = flit::setRandSeq<T>(n, seed);
  auto actual = flit::setRandSeq<T>(n, seed);
  for (decltype(n) i = 0; i < n; i++) {
    TH_EQUAL(expected[i], actual[i]);
    TH_VERIFY(expected[i] <= 6.0 && expected[i] >= -6.0);
  }

  // Changing the seed should give you a different sequence
  actual = flit::setRandSeq<T>(n, seed + 2);
  for (decltype(n) i = 0; i < n; i++) {
    TH_NOT_EQUAL(expected[i], actual[i]);
  }
}
void tst_setRandSeq_float() { tst_setRandSeq<float>(); }
void tst_setRandSeq_double() { tst_setRandSeq<double>(); }
void tst_setRandSeq_longdouble() { tst_setRandSeq<long double>(); }
TH_REGISTER(tst_setRandSeq_float);
TH_REGISTER(tst_setRandSeq_double);
TH_REGISTER(tst_setRandSeq_longdouble);

void tst_as_float_32bit() {
  uint32_t val = 1067316150;
  float expected = 1.234;
  TH_EQUAL(flit::as_float(val), expected);

  val = 0;
  expected = 0.0;
  TH_EQUAL(flit::as_float(val), expected);

  val = 1234;
  expected = 1.7292e-42;
  TH_EQUAL(flit::as_float(val), expected);
}
TH_REGISTER(tst_as_float_32bit);

void tst_as_float_64bit() {
  uint64_t val = 1067316150;
  double expected = 5.27324243e-315;
  TH_EQUAL(flit::as_float(val), expected);

  val = 0x3ff3be76c8b43958;
  expected = 1.234;
  TH_EQUAL(flit::as_float(val), expected);

  val = 0;
  expected = 0.0;
  TH_EQUAL(flit::as_float(val), expected);

  val = 1234;
  expected = 6.097e-321;
  TH_EQUAL(flit::as_float(val), expected);
}
TH_REGISTER(tst_as_float_64bit);

void tst_as_float_80bit() {
  auto val = combine_to_128(0x0000, 0x0000000000000000);
  long double expected = 0.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x3fff, 0x8000000000000000);
  expected = 1.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4000, 0x8000000000000000);
  expected = 2.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4000, 0xc000000000000000);
  expected = 3.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4001, 0x8000000000000000);
  expected = 4.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4001, 0xa000000000000000);
  expected = 5.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4001, 0xc000000000000000);
  expected = 6.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4001, 0xe000000000000000);
  expected = 7.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x4002, 0x8000000000000000);
  expected = 8.0L;
  TH_EQUAL(flit::as_float(val), expected);

  val = combine_to_128(0x2b97, 0xaed3412a32403b66);
  expected = 3.586714e-1573L;
  TH_EQUAL(flit::as_float(val), expected);
}
TH_REGISTER(tst_as_float_80bit);

void tst_as_int_32bit() {
  uint32_t expected = 1067316150;
  float val= 1.234;
  TH_EQUAL(flit::as_int(val), expected);

  expected = 0;
  val = 0.0;
  TH_EQUAL(flit::as_int(val), expected);

  expected = 1234;
  val = 1.7292e-42;
  TH_EQUAL(flit::as_int(val), expected);
}
TH_REGISTER(tst_as_int_32bit);

void tst_as_int_64bit() {
  uint64_t expected = 1067316150;
  double val = 5.27324243e-315;
  TH_EQUAL(flit::as_int(val), expected);

  expected = 0x3ff3be76c8b43958;
  val = 1.234;
  TH_EQUAL(flit::as_int(val), expected);

  expected = 0;
  val = 0.0;
  TH_EQUAL(flit::as_int(val), expected);

  expected = 1234;
  val = 6.097e-321;
  TH_EQUAL(flit::as_int(val), expected);
}
TH_REGISTER(tst_as_int_64bit);

void tst_as_int_128bit() {
  auto expected = combine_to_128(0x0000, 0x0000000000000000);
  long double val = 0.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x3fff, 0x8000000000000000);
  val = 1.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4000, 0x8000000000000000);
  val = 2.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4000, 0xc000000000000000);
  val = 3.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4001, 0x8000000000000000);
  val = 4.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4001, 0xa000000000000000);
  val = 5.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4001, 0xc000000000000000);
  val = 6.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4001, 0xe000000000000000);
  val = 7.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x4002, 0x8000000000000000);
  val = 8.0L;
  TH_EQUAL(flit::as_int(val), expected);

  expected = combine_to_128(0x2b97, 0xaed3412a32403b66);
  val = 3.586714e-1573L;
  TH_EQUAL(flit::as_int(val), expected);
}
TH_REGISTER(tst_as_int_128bit);
