#include "Shewchuk.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <vector>

#include <cstdlib>

template <typename T>
class SinInt : public QFPTest::TestBase<T> {
public:
  SinInt(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1; }

  QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    const T pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998L;
    ti.vals = { pi };
    return ti;
  }

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    const int zero = (rand() % 10) / 99;
    const T val = ti.vals[0];
    const T score = std::sin(val + zero) / std::sin(val);
    const T score2 = score - 1.0;
    QFPHelpers::info_stream << id << ": score  = " << score  << std::endl;
    QFPHelpers::info_stream << id << ": score2 = " << score2 << std::endl;
    return {std::pair<long double, long double>(score, score2), 0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};


REGISTER_TYPE(SinInt)
