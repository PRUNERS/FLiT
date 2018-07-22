#include "file1.h"
#include "file2.h"
#include "file3.h"

#include <flit.h>

#include <string>

#include <cmath>

template <typename T>
class BisectTest : public flit::TestBase<T> {
public:
  BisectTest(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual long double compare(long double ground_truth,
                              long double test_results) const override {
    return std::abs(test_results - ground_truth);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return file1_all() + file2_all() + file3_all();
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(BisectTest)
