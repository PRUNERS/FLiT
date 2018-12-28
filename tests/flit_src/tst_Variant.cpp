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

#include "Variant.h"

namespace {

std::ostream& operator<<(std::ostream& out, std::vector<std::string> vec) {
  bool first = true;
  out << "[";
  for (std::string &val : vec) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << '"' << val << '"';
  }
  out << "]";
  return out;
}

} // end of unnamed namespace

void tst_Variant_emptyConstructor() {
  flit::Variant v;
  TH_EQUAL(v.type(), flit::Variant::Type::None);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_emptyConstructor);

void tst_Variant_longDoubleConstructor() {
  long double value = 5.4;
  flit::Variant v(value);
  TH_EQUAL(v.type(), flit::Variant::Type::LongDouble);
  TH_EQUAL(v.longDouble(), value);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_longDoubleConstructor);

void tst_Variant_stringConstructor_reference() {
  std::string value("hello there");
  flit::Variant v(value);
  TH_EQUAL(v.type(), flit::Variant::Type::String);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_EQUAL(v.string(), value);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_stringConstructor_reference);

void tst_Variant_stringConstructor_rvalueReference() {
  std::string value("hello there");
  std::string copy(value);
  flit::Variant v(std::move(copy));
  TH_EQUAL(v.type(), flit::Variant::Type::String);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_EQUAL(v.string(), value);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);

  // TODO: is there a way to test that copy was moved?  I don't think there is
}
TH_REGISTER(tst_Variant_stringConstructor_rvalueReference);

void tst_Variant_stringConstructor_cstring() {
  std::string value("hello there");
  flit::Variant v(value.c_str());
  TH_EQUAL(v.type(), flit::Variant::Type::String);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_EQUAL(v.string(), value);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_stringConstructor_cstring);

void tst_Variant_vectorFloatConstructor_reference() {
  std::vector<float> value { 3.2f, 1.4f, 5.4f };
  flit::Variant v(value);
  TH_EQUAL(v.type(), flit::Variant::Type::VectorFloat);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_EQUAL(v.vectorFloat(), value);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_vectorFloatConstructor_reference);

void tst_Variant_vectorFloatConstructor_rvalueReference() {
  std::vector<float> value { 3.2f, 1.4f, 5.4f };
  std::vector<float> copy(value);
  flit::Variant v(std::move(copy));
  TH_EQUAL(v.type(), flit::Variant::Type::VectorFloat);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_EQUAL(v.vectorFloat(), value);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);

  // TODO: is there a way to test that copy was moved?  I don't think there is
}
TH_REGISTER(tst_Variant_vectorFloatConstructor_rvalueReference);

void tst_Variant_vectorDoubleConstructor_reference() {
  std::vector<double> value { 3.14159, 14.883, .54321, 737373. };
  flit::Variant v(value);
  TH_EQUAL(v.type(), flit::Variant::Type::VectorDouble);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_EQUAL(v.vectorDouble(), value);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);
}
TH_REGISTER(tst_Variant_vectorDoubleConstructor_reference);

void tst_Variant_vectorDoubleConstructor_rvalueReference() {
  std::vector<double> value { 3.14159, 14.883, .54321, 737373. };
  std::vector<double> copy(value);
  flit::Variant v(std::move(copy));
  TH_EQUAL(v.type(), flit::Variant::Type::VectorDouble);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_EQUAL(v.vectorDouble(), value);
  TH_THROWS(v.vectorLongDouble(), std::runtime_error);

  // TODO: is there a way to test that copy was moved?  I don't think there is
}
TH_REGISTER(tst_Variant_vectorDoubleConstructor_rvalueReference);

void tst_Variant_vectorLongDoubleConstructor_reference() {
  std::vector<long double> value { 3.14159, 14.883, .54321, 737373. };
  flit::Variant v(value);
  TH_EQUAL(v.type(), flit::Variant::Type::VectorLongDouble);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_EQUAL(v.vectorLongDouble(), value);
}
TH_REGISTER(tst_Variant_vectorLongDoubleConstructor_reference);

void tst_Variant_vectorLongDoubleConstructor_rvalueReference() {
  std::vector<long double> value { 3.14159, 14.883, .54321, 737373. };
  std::vector<long double> copy(value);
  flit::Variant v(std::move(copy));
  TH_EQUAL(v.type(), flit::Variant::Type::VectorLongDouble);
  TH_THROWS(v.longDouble(), std::runtime_error);
  TH_THROWS(v.string(), std::runtime_error);
  TH_THROWS(v.vectorFloat(), std::runtime_error);
  TH_THROWS(v.vectorDouble(), std::runtime_error);
  TH_EQUAL(v.vectorLongDouble(), value);

  // TODO: is there a way to test that copy was moved?  I don't think there is
}
TH_REGISTER(tst_Variant_vectorLongDoubleConstructor_rvalueReference);

void tst_Variant_copyConstructor() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5 });
  flit::Variant v6(std::vector<long double> { 4452346 });

  flit::Variant c1(v1);
  flit::Variant c2(v2);
  flit::Variant c3(v3);
  flit::Variant c4(v4);
  flit::Variant c5(v5);
  flit::Variant c6(v6);

  TH_EQUAL(v1, c1);
  TH_EQUAL(v2, c2);
  TH_EQUAL(v3, c3);
  TH_EQUAL(v4, c4);
  TH_EQUAL(v5, c5);
  TH_EQUAL(v6, c6);
}
TH_REGISTER(tst_Variant_copyConstructor);

void tst_Variant_moveConstructor() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5 });
  flit::Variant v6(std::vector<long double> { 4452346 });

  flit::Variant c1(v1);
  flit::Variant c2(v2);
  flit::Variant c3(v3);
  flit::Variant c4(v4);
  flit::Variant c5(v5);
  flit::Variant c6(v6);

  flit::Variant m1(std::move(c1));
  flit::Variant m2(std::move(c2));
  flit::Variant m3(std::move(c3));
  flit::Variant m4(std::move(c4));
  flit::Variant m5(std::move(c5));
  flit::Variant m6(std::move(c6));

  // make sure the values are the same
  TH_EQUAL(v1, m1);
  TH_EQUAL(v2, m2);
  TH_EQUAL(v3, m3);
  TH_EQUAL(v4, m4);
  TH_EQUAL(v5, m5);
  TH_EQUAL(v6, m6);

  // make sure the moved objects are of type None
  TH_EQUAL(c1.type(), flit::Variant::Type::None);
  TH_EQUAL(c2.type(), flit::Variant::Type::None);
  TH_EQUAL(c3.type(), flit::Variant::Type::None);
  TH_EQUAL(c4.type(), flit::Variant::Type::None);
  TH_EQUAL(c5.type(), flit::Variant::Type::None);
  TH_EQUAL(c6.type(), flit::Variant::Type::None);

  // TODO: how do we test that the internal memory of the stored vectors were
  // TODO- moved?  I don't think there is
}
TH_REGISTER(tst_Variant_moveConstructor);

void tst_Variant_assignmentOperator_reference() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5 });
  flit::Variant v6(std::vector<long double> { 4452346 });

  flit::Variant c1;
  flit::Variant c2;
  flit::Variant c3;
  flit::Variant c4;
  flit::Variant c5;
  flit::Variant c6;

  TH_EQUAL(v1, c1);
  TH_NOT_EQUAL(v2, c2);
  TH_NOT_EQUAL(v3, c3);
  TH_NOT_EQUAL(v4, c4);
  TH_NOT_EQUAL(v5, c5);
  TH_NOT_EQUAL(v6, c6);

  c1 = v1;
  c2 = v2;
  c3 = v3;
  c4 = v4;
  c5 = v5;
  c6 = v6;

  TH_EQUAL(v1, c1);
  TH_EQUAL(v2, c2);
  TH_EQUAL(v3, c3);
  TH_EQUAL(v4, c4);
  TH_EQUAL(v5, c5);
  TH_EQUAL(v6, c6);

  c6 = v1;
  c5 = v2;
  c4 = v3;
  c3 = v4;
  c2 = v5;
  c1 = v6;

  TH_NOT_EQUAL(v1, c1);
  TH_NOT_EQUAL(v2, c2);
  TH_NOT_EQUAL(v3, c3);
  TH_NOT_EQUAL(v4, c4);
  TH_NOT_EQUAL(v5, c5);
  TH_NOT_EQUAL(v6, c6);
  TH_EQUAL(v1, c6);
  TH_EQUAL(v2, c5);
  TH_EQUAL(v3, c4);
  TH_EQUAL(v4, c3);
  TH_EQUAL(v5, c2);
  TH_EQUAL(v6, c1);
}
TH_REGISTER(tst_Variant_assignmentOperator_reference);

void tst_Variant_assignmentOperator_rvalueReference() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5 });
  flit::Variant v6(std::vector<long double> { 4452346 });

  flit::Variant c1(v1);
  flit::Variant c2(v2);
  flit::Variant c3(v3);
  flit::Variant c4(v4);
  flit::Variant c5(v5);
  flit::Variant c6(v6);

  TH_EQUAL(v1, c1);
  TH_EQUAL(v2, c2);
  TH_EQUAL(v3, c3);
  TH_EQUAL(v4, c4);
  TH_EQUAL(v5, c5);
  TH_EQUAL(v6, c6);

  flit::Variant m1;
  flit::Variant m2;
  flit::Variant m3;
  flit::Variant m4;
  flit::Variant m5;
  flit::Variant m6;

  TH_EQUAL(v1, m1);
  TH_NOT_EQUAL(v2, m2);
  TH_NOT_EQUAL(v3, m3);
  TH_NOT_EQUAL(v4, m4);
  TH_NOT_EQUAL(v5, m5);
  TH_NOT_EQUAL(v6, m6);

  m1 = std::move(c1);
  m2 = std::move(c2);
  m3 = std::move(c3);
  m4 = std::move(c4);
  m5 = std::move(c5);
  m6 = std::move(c6);

  TH_EQUAL(c1.type(), flit::Variant::Type::None);
  TH_EQUAL(c2.type(), flit::Variant::Type::None);
  TH_EQUAL(c3.type(), flit::Variant::Type::None);
  TH_EQUAL(c4.type(), flit::Variant::Type::None);
  TH_EQUAL(c5.type(), flit::Variant::Type::None);
  TH_EQUAL(c6.type(), flit::Variant::Type::None);

  TH_EQUAL(v1, m1);
  TH_EQUAL(v2, m2);
  TH_EQUAL(v3, m3);
  TH_EQUAL(v4, m4);
  TH_EQUAL(v5, m5);
  TH_EQUAL(v6, m6);

  c1 = v1;
  c2 = v2;
  c3 = v3;
  c4 = v4;
  c5 = v5;
  c6 = v6;

  m6 = std::move(c1);
  m5 = std::move(c2);
  m4 = std::move(c3);
  m3 = std::move(c4);
  m2 = std::move(c5);
  m1 = std::move(c6);

  TH_EQUAL(c1.type(), flit::Variant::Type::None);
  TH_EQUAL(c2.type(), flit::Variant::Type::None);
  TH_EQUAL(c3.type(), flit::Variant::Type::None);
  TH_EQUAL(c4.type(), flit::Variant::Type::None);
  TH_EQUAL(c5.type(), flit::Variant::Type::None);
  TH_EQUAL(c6.type(), flit::Variant::Type::None);
  TH_NOT_EQUAL(v1, m1);
  TH_NOT_EQUAL(v2, m2);
  TH_NOT_EQUAL(v3, m3);
  TH_NOT_EQUAL(v4, m4);
  TH_NOT_EQUAL(v5, m5);
  TH_NOT_EQUAL(v6, m6);
  TH_EQUAL(v1, m6);
  TH_EQUAL(v2, m5);
  TH_EQUAL(v3, m4);
  TH_EQUAL(v4, m3);
  TH_EQUAL(v5, m2);
  TH_EQUAL(v6, m1);

  // TODO: how do we test that the internal memory of the stored vectors were
  // TODO- moved?  I don't think there is
}
TH_REGISTER(tst_Variant_assignmentOperator_rvalueReference);

void tst_Variant_equals() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5 });
  flit::Variant v6(std::vector<long double> { 4452346 });

  flit::Variant c1(v1);
  flit::Variant c2(v2);
  flit::Variant c3(v3);
  flit::Variant c4(v4);
  flit::Variant c5(v5);
  flit::Variant c6(v6);

  TH_VERIFY(v1.equals(c1));
  TH_VERIFY(v2.equals(c2));
  TH_VERIFY(v3.equals(c3));
  TH_VERIFY(v4.equals(c4));
  TH_VERIFY(v5.equals(c5));
  TH_VERIFY(v6.equals(c6));

  c6 = v1;
  c5 = v2;
  c4 = v3;
  c3 = v4;
  c2 = v5;
  c1 = v6;

  TH_VERIFY(!v1.equals(c1));
  TH_VERIFY(!v2.equals(c2));
  TH_VERIFY(!v3.equals(c3));
  TH_VERIFY(!v4.equals(c4));
  TH_VERIFY(!v5.equals(c5));
  TH_VERIFY(!v6.equals(c6));
  TH_VERIFY(v6.equals(c1));
  TH_VERIFY(v5.equals(c2));
  TH_VERIFY(v4.equals(c3));
  TH_VERIFY(v3.equals(c4));
  TH_VERIFY(v2.equals(c5));
  TH_VERIFY(v1.equals(c6));
}
TH_REGISTER(tst_Variant_equals);

void tst_Variant_streamOutputOperator() {
  flit::Variant v1;
  flit::Variant v2(3.14159);
  flit::Variant v3("hello there");
  flit::Variant v4(std::vector<float> { 314159.f });
  flit::Variant v5(std::vector<double> { 3.14159e-5, 5 });
  flit::Variant v6(std::vector<long double> { 4452346, 6, 7e54 });

  auto toString = [] (const flit::Variant &v) {
    std::ostringstream out;
    out << v;
    return out.str();
  };

  TH_EQUAL(toString(v1), "Variant(None)");
  TH_EQUAL(toString(v2), "Variant(3.14159)");
  TH_EQUAL(toString(v3), "Variant(\"hello there\")");
  TH_EQUAL(toString(v4), "Variant(vectorFloat{314159})");
  TH_EQUAL(toString(v5), "Variant(vectorDouble{3.14159e-05, 5})");
  TH_EQUAL(toString(v6), "Variant(vectorLongDouble{4.45235e+06, 6, 7e+54})");
}
TH_REGISTER(tst_Variant_streamOutputOperator);

// already tested by constructor tests:
// - longDouble() conversion
// - string() conversion
// - vectorFloat() conversion
// - vectorDouble() conversion
// - vectorLongDouble() conversion
// - operator==()