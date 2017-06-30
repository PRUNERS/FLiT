#include "Variant.h"

namespace flit {

std::ostream& operator<< (std::ostream& out, const Variant &val) {
  switch (val.type()) {
    case Variant::Type::None:
      out << "Variant(None)";
      break;
    case Variant::Type::LongDouble:
      out << "Variant(" << val.longDouble() << ")";
      break;
    case Variant::Type::String:
      out << "Variant(\"" << val.string() << "\")";
      break;
    default:
      throw std::runtime_error("Unimplemented type");
  }
  return out;
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
