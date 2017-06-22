#ifndef VARIANT_H
#define VARIANT_H

#include <stdexcept>
#include <string>

/** Can represent various different types
 *
 * This class is intented to be able to hold many different types in the
 * same object so that you can do things like make a list containing
 * sometimes strings and sometimes integers, etc.
 */
class Variant {
public:
  enum class Type {
    LongDouble = 1,
    String = 2,
  };

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

  Type type() { return _type; }

  long double longDouble() {
    if (_type != Type::LongDouble) {
      throw std::runtime_error("Variant is not of type Long Double");
    }
    return _ld_val;
  }

  std::string string() {
    if (_type != Type::String) {
      throw std::runtime_error("Variant is not of type String");
    }
    return _str_val;
  }

  template <typename T> T val();

private:
  Type _type;
  long double _ld_val { 0.0l };
  std::string _str_val { "" };
};

template <>
long double Variant::val() {
  return this->longDouble();
}

template <>
std::string Variant::val() {
  return this->string();
}

#endif // VARIANT_H
