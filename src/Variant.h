#ifndef VARIANT_H
#define VARIANT_H

#include <ostream>
#include <stdexcept>
#include <string>

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
      default:
        throw std::logic_error("Unimplemented Variant type in equals()");
    }
  }

private:
  Type _type;
  long double _ld_val { 0.0l };
  std::string _str_val { "" };
};

std::ostream& operator<< (std::ostream&, const Variant&);

inline bool operator== (const Variant& lhs, const Variant& rhs) {
  return lhs.equals(rhs);
}

} // end of namespace flit

#endif // VARIANT_H
