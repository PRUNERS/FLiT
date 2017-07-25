//this is the base instantiation for tests

#include "TestBase.h"

#include <stack>

namespace flit {

std::ostream& operator<<(std::ostream& os, const TestResult& res) {
  std::string comparison =
    (res.is_comparison_null() ? std::to_string(res.comparison()) : "NULL");

  os << res.name() << ":" << res.precision() << ","
     << res.result() << ","
     << comparison << ","
     << res.nanosecs();

  return os;
}

std::map<std::string, TestFactory*>& getTests() {
  static std::map<std::string, TestFactory*> tests;
  return tests;
}

} // end of namespace flit
