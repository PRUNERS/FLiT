#include "test_harness.h"

#include "flitHelpers.h"

#include "TestBase.h"   // for operator<<(flit::TestResult ...)

#include <algorithm>
#include <array>
#include <sstream>
#include <vector>

#include <cstdio>

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

// TODO: add tst_as_float_80bit()
// TODO: add tst_as_int_32bit()
// TODO: add tst_as_int_64bit()
// TODO: add tst_as_int_128bit()

