/**
 * Even though we try to make the test harness as simple as possible while
 * getting good functionality, we still should make sure it behaves as
 * expected.  This file uses the test harness and verifies its behaviors.
 *
 * Expected results are the following:
 *
 *   Failed tests:
 *     test_should_fail_1
 *     test_should_fail_2
 *     test_should_fail_3
 *     test_should_fail_4
 *     test_use_custom_equal_should_fail
 *     test_use_custom_not_equal_should_fail
 *
 *   Skipped tests:
 *     test_should_skip_1
 *     test_should_skip_2
 *
 *   Test Results:
 *     failures:   6
 *     successes:  3
 *     skips:      2
 *
 * with the expected return code of 6.
 */

#include "test_harness.h"

void test_a() {
  TH_VERIFY(1 + 2 == 3);
  TH_EQUAL(1 + 2, 3);
  TH_NOT_EQUAL(1 + 2, 4);
}
TH_REGISTER(test_a);

void test_should_fail_1() {
  TH_VERIFY(1 + 2 == 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_fail_1);

void test_should_fail_2() {
  TH_EQUAL(1 + 2, 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_fail_2);

void test_should_fail_3() {
  TH_FAIL("should fail");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_fail_3);

void fail_helper() {
  TH_FAIL("should fail");
}
void test_should_fail_4() {
  fail_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_fail_4);

void test_should_skip_1() {
  TH_SKIP("should skip");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_skip_1);

void skip_helper() {
  TH_SKIP("should skip");
}
void test_should_skip_2() {
  skip_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}
TH_REGISTER(test_should_skip_2);

struct Point { int x, y; };
bool operator==(const Point& a, const Point& b) {
  return a.y == b.y;
}
bool operator!=(const Point& a, const Point& b) {
  return a.x != b.x; // For whatever reason, different logic than for ==
}

void test_use_custom_equal() {
  Point a{1, 3};
  Point b{0, 3};
  TH_EQUAL(a, b);
  TH_VERIFY(a == b);
}
TH_REGISTER(test_use_custom_equal);

void test_use_custom_equal_should_fail() {
  Point a{0, 4};
  Point b{0, 3};
  TH_EQUAL(a, b);
}
TH_REGISTER(test_use_custom_equal_should_fail);

void test_use_custom_not_equal() {
  Point a{1, 3};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
  TH_VERIFY(a != b);
}
TH_REGISTER(test_use_custom_not_equal);

void test_use_custom_not_equal_should_fail() {
  Point a{0, 4};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
}
TH_REGISTER(test_use_custom_not_equal_should_fail);
