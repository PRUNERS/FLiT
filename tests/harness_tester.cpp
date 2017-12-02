/**
 * Even though we try to make the test harness as simple as possible while
 * getting good functionality, we still should make sure it behaves as
 * expected.  This file uses the test harness and verifies its behaviors.
 *
 * Expected results are the following:
 *
 *   Failed tests:
 *     tst_should_fail_1
 *     tst_should_fail_2
 *     tst_should_fail_3
 *     tst_should_fail_4
 *     tst_use_custom_equal_should_fail
 *     tst_use_custom_not_equal_should_fail
 *     tst_uncaught_exception_1
 *     tst_uncaught_exception_2
 *
 *   Skipped tests:
 *     tst_should_skip_1
 *     tst_should_skip_2
 *
 *   Test Results:
 *     failures:   8
 *     successes:  3
 *     skips:      2
 *
 * with the expected return code of 6.
 */

#include "test_harness.h"

void tst_a() {
  TH_VERIFY(1 + 2 == 3);
  TH_EQUAL(1 + 2, 3);
  TH_NOT_EQUAL(1 + 2, 4);
}
TH_REGISTER(tst_a);

void tst_should_fail_1() {
  TH_VERIFY(1 + 2 == 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_fail_1);

void tst_should_fail_2() {
  TH_EQUAL(1 + 2, 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_fail_2);

void tst_should_fail_3() {
  TH_FAIL("should fail");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_fail_3);

void fail_helper() {
  TH_FAIL("should fail");
}
void tst_should_fail_4() {
  fail_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_fail_4);

void tst_should_skip_1() {
  TH_SKIP("should skip");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_skip_1);

void skip_helper() {
  TH_SKIP("should skip");
}
void tst_should_skip_2() {
  skip_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_should_skip_2);

struct Point { int x, y; };
bool operator==(const Point& a, const Point& b) {
  return a.y == b.y;
}
bool operator!=(const Point& a, const Point& b) {
  return a.x != b.x; // For whatever reason, different logic than for ==
}

void tst_use_custom_equal() {
  Point a{1, 3};
  Point b{0, 3};
  TH_EQUAL(a, b);
  TH_VERIFY(a == b);
}
TH_REGISTER(tst_use_custom_equal);

void tst_use_custom_equal_should_fail() {
  Point a{0, 4};
  Point b{0, 3};
  TH_EQUAL(a, b);
}
TH_REGISTER(tst_use_custom_equal_should_fail);

void tst_use_custom_not_equal() {
  Point a{1, 3};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
  TH_VERIFY(a != b);
}
TH_REGISTER(tst_use_custom_not_equal);

void tst_use_custom_not_equal_should_fail() {
  Point a{0, 4};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
}
TH_REGISTER(tst_use_custom_not_equal_should_fail);

void tst_uncaught_exception_1() {
  throw std::logic_error("logical failure");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_uncaught_exception_1);

void tst_uncaught_exception_2() {
  throw 3;
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(tst_uncaught_exception_2);

