
#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <vector>

#include <cstdlib>

template <typename T>
class FMACancel : public QFPTest::TestBase<T> {
public:
  FMACancel(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 2; }

  QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.vals = { .1, 1.1e5 };
    return ti;
  }

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    const T a = ti.vals[0];
    const T b = ti.vals[1];
    const T c = a;
    const T d = -b;

    const T score = a*b + c*d;
    const T rtemp = c*d;
    const T score2 = a*b + rtemp;
    QFPHelpers::info_stream << id << ": score  = " << score  << std::endl;
    QFPHelpers::info_stream << id << ": score2 = " << score2 << std::endl;
    return {score, score2};
  }

protected:
  using QFPTest::TestBase<T>::id;
};


REGISTER_TYPE(FMACancel)
