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

#include "test_harness.h"

#include <flit/flitHelpers.h>
#include <flit/TestBase.h>   // for operator<<(flit::TestResult ...)

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstring>  // for memcpy()

namespace {

unsigned __int128 combine_to_128(uint64_t left_half, uint64_t right_half) {
  unsigned __int128 val = left_half;
  val = val << 64;
  val += right_half;
  return val;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T> &vec) {
  out << "[";
  bool first = true;
  for (auto &val : vec) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << val;
  }
  out << "]";
  return out;
}

} // end of unnamed namespace

namespace tst_as_float {

TH_TEST(tst_as_float_32bit) {
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

TH_TEST(tst_as_float_64bit) {
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

TH_TEST(tst_as_float_80bit) {
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

} // end of namespace tst_as_float

namespace tst_as_int {

TH_TEST(tst_as_int_32bit) {
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

TH_TEST(tst_as_int_64bit) {
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

TH_TEST(tst_as_int_128bit) {
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

} // end of namespace tst_as_int

namespace tst_abs_compare {

template <typename T>
void tst_equal_with_nan_inf_impl() {
  using lim = std::numeric_limits<T>;

  static_assert(lim::has_quiet_NaN, "Type T does not have quiet NaNs");
  static_assert(lim::has_infinity,  "Type T does not have infinity");

  auto &eq = flit::equal_with_nan_inf<T>;
  T my_nan = lim::quiet_NaN();
  T my_inf = lim::infinity();
  T normal = -3.2;
  T zero = 0.0;

  TH_VERIFY( eq( my_nan,  my_nan));
  TH_VERIFY(!eq( my_nan, -my_nan));
  TH_VERIFY(!eq( my_nan,  my_inf));
  TH_VERIFY(!eq( my_nan, -my_inf));
  TH_VERIFY(!eq( my_nan,  normal));
  TH_VERIFY(!eq( my_nan, -normal));

  TH_VERIFY(!eq(-my_nan,  my_nan));
  TH_VERIFY( eq(-my_nan, -my_nan));
  TH_VERIFY(!eq(-my_nan,  my_inf));
  TH_VERIFY(!eq(-my_nan, -my_inf));
  TH_VERIFY(!eq(-my_nan,  normal));
  TH_VERIFY(!eq(-my_nan, -normal));

  TH_VERIFY(!eq( my_inf,  my_nan));
  TH_VERIFY(!eq( my_inf, -my_nan));
  TH_VERIFY( eq( my_inf,  my_inf));
  TH_VERIFY(!eq( my_inf, -my_inf));
  TH_VERIFY(!eq( my_inf,  normal));
  TH_VERIFY(!eq( my_inf, -normal));

  TH_VERIFY(!eq(-my_inf,  my_nan));
  TH_VERIFY(!eq(-my_inf, -my_nan));
  TH_VERIFY(!eq(-my_inf,  my_inf));
  TH_VERIFY( eq(-my_inf, -my_inf));
  TH_VERIFY(!eq(-my_inf,  normal));
  TH_VERIFY(!eq(-my_inf, -normal));

  TH_VERIFY(!eq( normal,  my_nan));
  TH_VERIFY(!eq( normal, -my_nan));
  TH_VERIFY(!eq( normal,  my_inf));
  TH_VERIFY(!eq( normal, -my_inf));
  TH_VERIFY( eq( normal,  normal));
  TH_VERIFY(!eq( normal, -normal));

  TH_VERIFY(!eq(-normal,  my_nan));
  TH_VERIFY(!eq(-normal, -my_nan));
  TH_VERIFY(!eq(-normal,  my_inf));
  TH_VERIFY(!eq(-normal, -my_inf));
  TH_VERIFY(!eq(-normal,  normal));
  TH_VERIFY( eq(-normal, -normal));
}

TH_TEST(tst_equal_with_nan_inf) {
  tst_equal_with_nan_inf_impl<float>();
  tst_equal_with_nan_inf_impl<double>();
  tst_equal_with_nan_inf_impl<long double>();
}

template<typename T>
void tst_abs_compare_impl() {
  using lim = std::numeric_limits<T>;
  static_assert(lim::has_quiet_NaN, "Type T does not have quiet NaNs");
  static_assert(lim::has_infinity,  "Type T does not have infinity");
  T my_nan = lim::quiet_NaN();
  T my_inf = lim::infinity();
  T normal = -3.2;
  T zero = 0.0;

  auto &eq = flit::equal_with_nan_inf<T>;
  auto &comp = flit::abs_compare<T>;

  // we have 25 cases
  TH_VERIFY(eq(comp( my_nan,  my_nan),   zero ));
  TH_VERIFY(eq(comp( my_nan, -my_nan),  my_nan));
  TH_VERIFY(eq(comp( my_nan,  my_inf),  my_nan));
  TH_VERIFY(eq(comp( my_nan, -my_inf),  my_nan));
  TH_VERIFY(eq(comp( my_nan,  normal),  my_nan));

  TH_VERIFY(eq(comp(-my_nan,  my_nan),  my_nan));
  TH_VERIFY(eq(comp(-my_nan, -my_nan),   zero ));
  TH_VERIFY(eq(comp(-my_nan,  my_inf),  my_nan));
  TH_VERIFY(eq(comp(-my_nan, -my_inf),  my_nan));
  TH_VERIFY(eq(comp(-my_nan,  normal),  my_nan));

  TH_VERIFY(eq(comp( my_inf,  my_nan),  my_nan));
  TH_VERIFY(eq(comp( my_inf, -my_nan),  my_nan));
  TH_VERIFY(eq(comp( my_inf,  my_inf),   zero ));
  TH_VERIFY(eq(comp( my_inf, -my_inf),  my_inf));
  TH_VERIFY(eq(comp( my_inf,  normal),  my_inf));

  TH_VERIFY(eq(comp(-my_inf,  my_nan),  my_nan));
  TH_VERIFY(eq(comp(-my_inf, -my_nan),  my_nan));
  TH_VERIFY(eq(comp(-my_inf,  my_inf),  my_inf));
  TH_VERIFY(eq(comp(-my_inf, -my_inf),   zero ));
  TH_VERIFY(eq(comp(-my_inf,  normal),  my_inf));

  TH_VERIFY(eq(comp( normal,  my_nan),  my_nan));
  TH_VERIFY(eq(comp( normal, -my_nan),  my_nan));
  TH_VERIFY(eq(comp( normal,  my_inf),  my_inf));
  TH_VERIFY(eq(comp( normal, -my_inf),  my_inf));
  TH_VERIFY(eq(comp( normal,  normal),   zero ));
}

TH_TEST(tst_abs_compare) {
  tst_abs_compare_impl<float>();
  //tst_abs_compare_impl<double>();
  //tst_abs_compare_impl<long double>();
}

} // end of namespace tst_abs_compare

namespace tst_l2norm {

TH_TEST(tst_l2norm_normal_numbers) {
  std::vector<float> empty;
  std::vector<float> one_elem { 1.0L };
  std::vector<float> two_elems { 3.5L, 1.0L };

  TH_EQUAL(flit::l2norm(empty    , empty    ),  0.0L );
  TH_EQUAL(flit::l2norm(one_elem , empty    ),  1.0L );
  TH_EQUAL(flit::l2norm(empty    , one_elem ),  1.0L );
  TH_EQUAL(flit::l2norm(one_elem , one_elem ),  0.0L );
  TH_EQUAL(flit::l2norm(two_elems, empty    ), 13.25L);
  TH_EQUAL(flit::l2norm(two_elems, one_elem ),  7.25L);
  TH_EQUAL(flit::l2norm(empty    , two_elems), 13.25L);
  TH_EQUAL(flit::l2norm(one_elem , two_elems),  7.25L);
  TH_EQUAL(flit::l2norm(two_elems, two_elems),  0.0L );
}

TH_TEST(tst_l2norm_nan_and_inf) {
  using lim = std::numeric_limits<float>;
  float my_nan = lim::quiet_NaN();
  float my_inf = lim::infinity();

  auto &eq = flit::equal_with_nan_inf<long double>;

  std::vector<float> nans  { my_nan, -my_nan, my_nan};
  std::vector<float> nans2 {-my_nan, -my_nan, my_nan}; // mismatching in val 0
  std::vector<float> infs  { my_inf, -my_inf, my_inf};
  std::vector<float> infs2 { my_inf,  my_inf, my_inf}; // mismatching in val 1

  TH_VERIFY(eq(flit::l2norm(nans, nans ),  0.0L ));
  TH_VERIFY(eq(flit::l2norm(nans, nans2), my_nan));
  TH_VERIFY(eq(flit::l2norm(infs, infs ),  0.0L ));
  TH_VERIFY(eq(flit::l2norm(infs, infs2), my_inf));
  TH_VERIFY(eq(flit::l2norm(nans, infs ), my_nan));
}

} // end of namespace tst_l2norm

namespace tst_split {

TH_TEST(tst_split_empty) {
  TH_EQUAL(flit::split("", '\n'), std::vector<std::string>{});
}

TH_TEST(tst_split_nonempty) {
  std::string text("hello");
  auto actual = flit::split(text, ' ');
  std::vector<std::string> expected {text};
  TH_EQUAL(actual, expected);
}

TH_TEST(tst_split_one_split) {
  std::string text = "hello there";
  std::vector<std::string> expected_split = { "hello", "there" };
  TH_EQUAL(flit::split(text, ' '), expected_split);
}

TH_TEST(tst_split_many_splits) {
  std::string text = "hello there my friend";
  std::vector<std::string> expected_split {"hello", "there", "my", "friend"};
  TH_EQUAL(flit::split(text, ' '), expected_split);
  expected_split = {
    "h",
    "llo th",
    "r",
    " my fri",
    "nd",
  };
  TH_EQUAL(flit::split(text, 'e'), expected_split);
}

TH_TEST(tst_split_maxsplit_zero) {
  std::string text = "hello there my friend";
  std::vector<std::string> expected_split { "hello there my friend" };
  TH_EQUAL(flit::split(text, ' ', 0), expected_split);
}

TH_TEST(tst_split_maxsplit_one) {
  std::string text = "hello there my friend";
  std::vector<std::string> expected_split { "hello", "there my friend" };
  TH_EQUAL(flit::split(text, ' ', 1), expected_split);
}

TH_TEST(tst_split_maxsplit_many) {
  std::string text = "hello there my friend";
  std::vector<std::string> expected_split { "hello", "there", "my", "friend" };
  TH_EQUAL(flit::split(text, ' ', 100), expected_split);
  TH_EQUAL(flit::split(text, ' ', 3), expected_split);

  expected_split = { "hello", "there", "my friend" };
  TH_EQUAL(flit::split(text, ' ', 2), expected_split);
}

} // end of namespace tst_split

namespace tst_trim {

TH_TEST(tst_trim_empty) {
  TH_EQUAL(flit::trim(""), "");
}

TH_TEST(tst_trim_nothing_to_trim) {
  TH_EQUAL(flit::trim("hello"), "hello");
}

TH_TEST(tst_trim_left_spaces) {
  TH_EQUAL(flit::trim("  monty"), "monty");
}

TH_TEST(tst_trim_right_spaces) {
  TH_EQUAL(flit::trim("python  "), "python");
}

TH_TEST(tst_trim_leftright_spaces) {
  TH_EQUAL(flit::trim("  ROCKS!!  "), "ROCKS!!");
}

TH_TEST(tst_trim_tabs) {
  TH_EQUAL(flit::trim("\ttabs-suck!\t\t"), "tabs-suck!");
}

TH_TEST(tst_trim_newlines) {
  TH_EQUAL(flit::trim("\nmike-is-cool\n\n"), "mike-is-cool");
}

TH_TEST(tst_trim_formfeed) {
  TH_EQUAL(flit::trim("\f\fwho-uses-form-feeds?\f\f\f"),
           "who-uses-form-feeds?");
}

TH_TEST(tst_trim_carriagereturn) {
  TH_EQUAL(flit::trim("\r\rwho-uses-carriage-returns?\r\r"),
           "who-uses-carriage-returns?");
}

TH_TEST(tst_trim_verticaltab) {
  TH_EQUAL(flit::trim("\v\vvertical-tabs?--really?\v\v"),
           "vertical-tabs?--really?");
}

TH_TEST(tst_trim_alltypes) {
  TH_EQUAL(flit::trim(" \t\n\f\r\vall-types \t\f\n\v\r"),
           "all-types");
}

TH_TEST(tst_trim_onlywhitespace) {
  TH_EQUAL(flit::trim("  \t\n \f \r\v \r   \t\n \n"), "");
}

TH_TEST(tst_trim_whitespace_inside) {
  TH_EQUAL(flit::trim(" whitespace\tis\ninside of this\rphrase\n  "),
           "whitespace\tis\ninside of this\rphrase");
}

TH_TEST(tst_ltrim_empty) {
  TH_EQUAL(flit::ltrim(""), "");
}

TH_TEST(tst_ltrim_nothing_to_ltrim) {
  TH_EQUAL(flit::ltrim("hello"), "hello");
}

TH_TEST(tst_ltrim_left_spaces) {
  TH_EQUAL(flit::ltrim("  monty"), "monty");
}

TH_TEST(tst_ltrim_right_spaces) {
  TH_EQUAL(flit::ltrim("python  "), "python  ");
}

TH_TEST(tst_ltrim_leftright_spaces) {
  TH_EQUAL(flit::ltrim("  ROCKS!!  "), "ROCKS!!  ");
}

TH_TEST(tst_ltrim_tabs) {
  TH_EQUAL(flit::ltrim("\ttabs-suck!\t\t"), "tabs-suck!\t\t");
}

TH_TEST(tst_ltrim_newlines) {
  TH_EQUAL(flit::ltrim("\nmike-is-cool\n\n"), "mike-is-cool\n\n");
}

TH_TEST(tst_ltrim_formfeed) {
  TH_EQUAL(flit::ltrim("\f\fwho-uses-form-feeds?\f\f\f"),
           "who-uses-form-feeds?\f\f\f");
}

TH_TEST(tst_ltrim_carriagereturn) {
  TH_EQUAL(flit::ltrim("\r\rwho-uses-carriage-returns?\r\r"),
           "who-uses-carriage-returns?\r\r");
}

TH_TEST(tst_ltrim_verticaltab) {
  TH_EQUAL(flit::ltrim("\v\vvertical-tabs?--really?\v\v"),
           "vertical-tabs?--really?\v\v");
}

TH_TEST(tst_ltrim_alltypes) {
  TH_EQUAL(flit::ltrim(" \t\n\f\r\vall-types \t\f\n\v\r"),
           "all-types \t\f\n\v\r");
}

TH_TEST(tst_ltrim_onlywhitespace) {
  TH_EQUAL(flit::ltrim("  \t\n \f \r\v \r   \t\n \n"), "");
}

TH_TEST(tst_ltrim_whitespace_inside) {
  TH_EQUAL(flit::ltrim(" whitespace\tis\ninside of this\rphrase\n  "),
           "whitespace\tis\ninside of this\rphrase\n  ");
}

TH_TEST(tst_rtrim_empty) {
  TH_EQUAL(flit::rtrim(""), "");
}

TH_TEST(tst_rtrim_nothing_to_rtrim) {
  TH_EQUAL(flit::rtrim("hello"), "hello");
}

TH_TEST(tst_rtrim_left_spaces) {
  TH_EQUAL(flit::rtrim("  monty"), "  monty");
}

TH_TEST(tst_rtrim_right_spaces) {
  TH_EQUAL(flit::rtrim("python  "), "python");
}

TH_TEST(tst_rtrim_leftright_spaces) {
  TH_EQUAL(flit::rtrim("  ROCKS!!  "), "  ROCKS!!");
}

TH_TEST(tst_rtrim_tabs) {
  TH_EQUAL(flit::rtrim("\ttabs-suck!\t\t"), "\ttabs-suck!");
}

TH_TEST(tst_rtrim_newlines) {
  TH_EQUAL(flit::rtrim("\nmike-is-cool\n\n"), "\nmike-is-cool");
}

TH_TEST(tst_rtrim_formfeed) {
  TH_EQUAL(flit::rtrim("\f\fwho-uses-form-feeds?\f\f\f"),
           "\f\fwho-uses-form-feeds?");
}

TH_TEST(tst_rtrim_carriagereturn) {
  TH_EQUAL(flit::rtrim("\r\rwho-uses-carriage-returns?\r\r"),
           "\r\rwho-uses-carriage-returns?");
}

TH_TEST(tst_rtrim_verticaltab) {
  TH_EQUAL(flit::rtrim("\v\vvertical-tabs?--really?\v\v"),
           "\v\vvertical-tabs?--really?");
}

TH_TEST(tst_rtrim_alltypes) {
  TH_EQUAL(flit::rtrim(" \t\n\f\r\vall-types \t\f\n\v\r"),
           " \t\n\f\r\vall-types");
}

TH_TEST(tst_rtrim_onlywhitespace) {
  TH_EQUAL(flit::rtrim("  \t\n \f \r\v \r   \t\n \n"), "");
}

TH_TEST(tst_rtrim_whitespace_inside) {
  TH_EQUAL(flit::rtrim(" whitespace\tis\ninside of this\rphrase\n  "),
           " whitespace\tis\ninside of this\rphrase");
}

} // end of namespace tst_trim

namespace tst_strip {

TH_TEST(tst_rstrip_empty) {
  TH_EQUAL(flit::rstrip("", "/"), "");
  TH_EQUAL(flit::rstrip("", ""), "");
  TH_EQUAL(flit::rstrip("...", ""), "...");
}

TH_TEST(tst_rstrip_nothing_to_strip) {
  TH_EQUAL(flit::rstrip("hello", " "), "hello");
}

TH_TEST(tst_rstrip_bothsides) {
  TH_EQUAL(flit::rstrip("--hello-there--", "-"), "--hello-there");
}

TH_TEST(tst_rstrip_multichar) {
  TH_EQUAL(flit::rstrip("/*/*/* /* hi there /* /*/*/*", "/*"),
           "/*/*/* /* hi there /* ");
}

TH_TEST(tst_lstrip_empty) {
  TH_EQUAL(flit::lstrip("", "/"), "");
  TH_EQUAL(flit::lstrip("", ""), "");
  TH_EQUAL(flit::lstrip("...", ""), "...");
}

TH_TEST(tst_lstrip_nothing_to_strip) {
  TH_EQUAL(flit::lstrip("hello", " "), "hello");
}

TH_TEST(tst_lstrip_bothsides) {
  TH_EQUAL(flit::lstrip("--hello-there--", "-"), "hello-there--");
}

TH_TEST(tst_lstrip_multichar) {
  TH_EQUAL(flit::lstrip("/*/*/* /* hi there /* /*/*/*", "/*"),
           " /* hi there /* /*/*/*");
}

TH_TEST(tst_strip_empty) {
  TH_EQUAL(flit::strip("", "/"), "");
  TH_EQUAL(flit::strip("", ""), "");
  TH_EQUAL(flit::strip("...", ""), "...");
}

TH_TEST(tst_strip_nothing_to_strip) {
  TH_EQUAL(flit::strip("hello", " "), "hello");
}

TH_TEST(tst_strip_bothsides) {
  TH_EQUAL(flit::strip("--hello-there--", "-"), "hello-there");
}

TH_TEST(tst_strip_multichar) {
  TH_EQUAL(flit::strip("/*/*/* /* hi there /* /*/*/*", "/*"),
           " /* hi there /* ");
}

} // end of namespace tst_strip
