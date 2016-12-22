#include "Kahan.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <string>

template <typename T>
class KahanSum : public QFPTest::TestBase<T> {
public:
  KahanSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Kahan<T> kahan;
    for (auto val : ti.vals) {
      kahan.add(val);
    }
    return {kahan.sum(), 0.0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

