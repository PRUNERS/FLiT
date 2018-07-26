/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
