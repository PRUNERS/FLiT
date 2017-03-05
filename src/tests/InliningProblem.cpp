
#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <vector>

#include <cstdlib>

template <typename T>
class InliningProblem : public QFPTest::TestBase<T> {
public:
  InliningProblem(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1; }

  QFPTest::TestInput<T> getDefaultInput() {
    QFPTest::TestInput<T> ti;
    ti.vals = { .1, 1.1e3, -.1, -1.1e3, 1/3 };
    return ti;
  }

protected:
  T identity(const T x) {
    const T nx = -x;
    const T x_again = -nx;
    return x_again;
  }
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    T a = ti.vals[0];
    T also_a = identity(a);

    const T score = std::sqrt(a) * std::sqrt(also_a);
    const T score2 = std::pow(std::sqrt(a), 2);

    QFPHelpers::info_stream << id << ": score  = " << score  << std::endl;
    QFPHelpers::info_stream << id << ": score2 = " << score2 << std::endl;
    return {std::pair<long double, long double>(score, score2), 0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};


REGISTER_TYPE(InliningProblem)
