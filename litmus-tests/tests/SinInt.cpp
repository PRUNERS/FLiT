#include "Shewchuk.h"

#include <flit.h>

#include <vector>

#include <cstdlib>

template <typename T>
class SinInt : public flit::TestBase<T> {
public:
  SinInt(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 1; }

  virtual flit::TestInput<T> getDefaultInput() override {
    flit::TestInput<T> ti;
    const T pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998L;
    ti.vals = { pi };
    return ti;
  }

protected:
  virtual flit::Variant run_impl(const flit::TestInput<T>& ti) override {
    const int zero = (rand() % 10) / 99;
    const T val = ti.vals[0];
    const T score = std::sin(val + zero) / std::sin(val);
    flit::info_stream << id << ": score       = " << score       << std::endl;
    flit::info_stream << id << ": score - 1.0 = " << score - 1.0 << std::endl;
    return score;
  }

protected:
  using flit::TestBase<T>::id;
};


REGISTER_TYPE(SinInt)
