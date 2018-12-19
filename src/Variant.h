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

#ifndef VARIANT_H
#define VARIANT_H

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flit {

/** Can represent various different types
 *
 * This class is intented to be able to hold many different types in the
 * same object so that you can do things like make a list containing
 * sometimes strings and sometimes integers, etc.
 */
class Variant {
public:
  enum class Type {
    None = 1,
    LongDouble = 2,
    String = 3,
    VectorFloat = 4,
    VectorDouble = 5,
    VectorLongDouble = 6,
  };

  Variant() : _type(Type::None) { }

  Variant(long double val)
    : _type(Type::LongDouble)
    , _ld_val(val) { }

  Variant(std::string &val)
    : _type(Type::String)
    , _str_val(val) { }
  Variant(const std::string &val)
    : _type(Type::String)
    , _str_val(val) { }
  Variant(std::string &&val)
    : _type(Type::String)
    , _str_val(val) { }
  Variant(const char* val)
    : _type(Type::String)
    , _str_val(val) { }

  Variant(std::vector<float> &val)
    : _type(Type::VectorFloat)
    , _vecflt_val(val) { }
  Variant(const std::vector<float> &val)
    : _type(Type::VectorFloat)
    , _vecflt_val(val) { }
  Variant(std::vector<float> &&val)
    : _type(Type::VectorFloat)
    , _vecflt_val(val) { }

  Variant(std::vector<double> &val)
    : _type(Type::VectorDouble)
    , _vecdbl_val(val) { }
  Variant(const std::vector<double> &val)
    : _type(Type::VectorDouble)
    , _vecdbl_val(val) { }
  Variant(std::vector<double> &&val)
    : _type(Type::VectorDouble)
    , _vecdbl_val(val) { }

  Variant(std::vector<long double> &val)
    : _type(Type::VectorLongDouble)
    , _vecldbl_val(val) { }
  Variant(const std::vector<long double> &val)
    : _type(Type::VectorLongDouble)
    , _vecldbl_val(val) { }
  Variant(std::vector<long double> &&val)
    : _type(Type::VectorLongDouble)
    , _vecldbl_val(val) { }

  Variant(const Variant &other)
    : _type(other._type)
  {
    *this = other;
  }

  Variant(Variant &&other)
    : _type(other._type)
    , _ld_val(other._ld_val)
    , _str_val(std::move(other._str_val))
    , _vecflt_val(std::move(other._vecflt_val))
    , _vecdbl_val(std::move(other._vecdbl_val))
    , _vecldbl_val(std::move(other._vecldbl_val)) { }

  Variant& operator=(const Variant &other) {
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

  bool operator==(const Variant& other) {
    return this->equals(other);
  }

  Type type() const { return _type; }

  long double longDouble() const {
    if (_type != Type::LongDouble) {
      throw std::runtime_error("Variant is not of type Long Double");
    }
    return _ld_val;
  }

  std::string string() const {
    if (_type != Type::String) {
      throw std::runtime_error("Variant is not of type String");
    }
    return _str_val;
  }

  std::vector<float> vectorFloat() const {
    if (_type != Type::VectorFloat) {
      throw std::runtime_error("Variant is not of type std::vector<float>");
    }
    return _vecflt_val;
  }

  std::vector<double> vectorDouble() const {
    if (_type != Type::VectorDouble) {
      throw std::runtime_error("Variant is not of type std::vector<double>");
    }
    return _vecdbl_val;
  }

  std::vector<long double> vectorLongDouble() const {
    if (_type != Type::VectorLongDouble) {
      throw std::runtime_error(
          "Variant is not of type std::vector<long double>");
    }
    return _vecldbl_val;
  }

  template <typename T> T val() const;

  bool equals(const Variant &other) const {
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

private:
  Type _type;
  long double _ld_val { 0.0l };
  std::string _str_val { "" };
  std::vector<float> _vecflt_val { };
  std::vector<double> _vecdbl_val { };
  std::vector<long double> _vecldbl_val { };
};

std::ostream& operator<< (std::ostream&, const Variant&);

inline bool operator== (const Variant& lhs, const Variant& rhs) {
  return lhs.equals(rhs);
}

} // end of namespace flit

#endif // VARIANT_H
