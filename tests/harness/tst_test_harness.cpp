/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

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

TH_TEST(tst_a) {
  TH_VERIFY(1 + 2 == 3);
  TH_EQUAL(1 + 2, 3);
  TH_NOT_EQUAL(1 + 2, 4);
}

TH_TEST(tst_should_fail_1) {
  TH_VERIFY(1 + 2 == 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

TH_TEST(tst_should_fail_2) {
  TH_EQUAL(1 + 2, 4);
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

TH_TEST(tst_should_fail_3) {
  TH_FAIL("should fail");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

void fail_helper() {
  TH_FAIL("should fail");
}
TH_TEST(tst_should_fail_4) {
  fail_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

TH_TEST(tst_should_skip_1) {
  TH_SKIP("should skip");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

void skip_helper() {
  TH_SKIP("should skip");
}
TH_TEST(tst_should_skip_2) {
  skip_helper();
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

struct Point { int x, y; };
bool operator==(const Point& a, const Point& b) {
  return a.y == b.y;
}
bool operator!=(const Point& a, const Point& b) {
  return a.x != b.x; // For whatever reason, different logic than for ==
}

TH_TEST(tst_use_custom_equal) {
  Point a{1, 3};
  Point b{0, 3};
  TH_EQUAL(a, b);
  TH_VERIFY(a == b);
}

TH_TEST(tst_use_custom_equal_should_fail) {
  Point a{0, 4};
  Point b{0, 3};
  TH_EQUAL(a, b);
}

TH_TEST(tst_use_custom_not_equal) {
  Point a{1, 3};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
  TH_VERIFY(a != b);
}

TH_TEST(tst_use_custom_not_equal_should_fail) {
  Point a{0, 4};
  Point b{0, 3};
  TH_NOT_EQUAL(a, b);
}

TH_TEST(tst_uncaught_exception_1) {
  throw std::logic_error("logical failure");
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

TH_TEST(tst_uncaught_exception_2) {
  throw 3;
  std::cout << "YOU SHOULD NEVER SEE THIS MESSAGE!\n";
}

TH_TEST(tst_expected_throw_works_1) {
  TH_THROWS(throw std::logic_error("logical failure"), std::logic_error);
}

void throw_helper() {
  throw std::runtime_error("runtime failure");
}
TH_TEST(tst_expected_throw_works_2) {
  TH_THROWS(throw_helper(), std::runtime_error);
}

TH_TEST(tst_expected_throw_works_3) {
  TH_THROWS(throw 3, int);
}

TH_TEST(tst_expected_throw_fails_1) {
  TH_THROWS(throw std::logic_error("logic failure msg"), std::runtime_error);
}

TH_TEST(tst_expected_throw_fails_2) {
  TH_THROWS(throw 1.2, float);
}

