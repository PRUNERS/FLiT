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

#include "Variant.h"

#include <sstream>

namespace {

template <typename T>
std::ostream& vecToStream(std::ostream& out, std::vector<T> vec) {
  out << "{";
  bool first = true;
  for (auto item : vec) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << item;
  }
  out << "}";
  return out;
}

} // end of unnamed namespace

namespace flit {

Variant& Variant::operator=(const Variant &other) {
  _type = other._type;
  switch (_type) {
    case Type::None:
      break;
    case Type::LongDouble:
      _ld_val = other._ld_val;
      break;
    case Type::String:
      _str_val = other._str_val;
      break;
    case Type::VectorFloat:
      _vecflt_val = other._vecflt_val;
      break;
    case Type::VectorDouble:
      _vecdbl_val = other._vecdbl_val;
      break;
    case Type::VectorLongDouble:
      _vecldbl_val = other._vecldbl_val;
      break;
    default:
      throw std::logic_error(
          "Unimplemented Variant type in assignment operator");
  }
  return *this;
}

Variant& Variant::operator=(Variant &&other) {
  _type = other._type;
  other._type = Type::None;
  switch (_type) {
    case Type::None:
      break;
    case Type::LongDouble:
      _ld_val = std::move(other._ld_val);
      break;
    case Type::String:
      _str_val = std::move(other._str_val);
      break;
    case Type::VectorFloat:
      _vecflt_val = std::move(other._vecflt_val);
      break;
    case Type::VectorDouble:
      _vecdbl_val = std::move(other._vecdbl_val);
      break;
    case Type::VectorLongDouble:
      _vecldbl_val = std::move(other._vecldbl_val);
      break;
    default:
      throw std::logic_error(
          "Unimplemented Variant type in move assignment operator");
  }
  return *this;
}

bool Variant::equals(const Variant &other) const {
  if (_type != other._type) {
    return false;
  }
  switch (_type) {
    case Type::None:
      return true;
    case Type::LongDouble:
      return _ld_val == other._ld_val;
    case Type::String:
      return _str_val == other._str_val;
    case Type::VectorFloat:
      return _vecflt_val == other._vecflt_val;
    case Type::VectorDouble:
      return _vecdbl_val == other._vecdbl_val;
    case Type::VectorLongDouble:
      return _vecldbl_val == other._vecldbl_val;
    default:
      throw std::logic_error("Unimplemented Variant type in equals()");
  }
}

std::string Variant::toString() const {
  std::ostringstream out;
  switch (type()) {
    case Variant::Type::None:
      out << "Variant(None)";
      break;
    case Variant::Type::LongDouble:
      out << "Variant(" << longDouble() << ")";
      break;
    case Variant::Type::String:
      out << "Variant(\"" << string() << "\")";
      break;
    case Variant::Type::VectorFloat:
      out << "Variant(vectorFloat";
      vecToStream(out, vectorFloat());
      out << ")";
      break;
    case Variant::Type::VectorDouble:
      out << "Variant(vectorDouble";
      vecToStream(out, vectorDouble());
      out << ")";
      break;
    case Variant::Type::VectorLongDouble:
      out << "Variant(vectorLongDouble";
      vecToStream(out, vectorLongDouble());
      out << ")";
      break;
    default:
      throw std::runtime_error("Unimplemented type");
  }
  return out.str();
}

template <>
long double Variant::val() const {
  return this->longDouble();
}

template <>
std::string Variant::val() const {
  return this->string();
}

} // end of namespace flit
