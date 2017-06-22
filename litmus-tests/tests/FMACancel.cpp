#include <flit.h>

#include <vector>

#include <cstdlib>

template <typename T>
class FMACancel : public flit::TestBase<T> {
public:
  FMACancel(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 2; }

  flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = { .1, 1.1e5 };
    return ti;
  }

protected:
  virtual long double run_impl(const flit::TestInput<T>& ti) {
    const T a = ti.vals[0];
    const T b = ti.vals[1];
    const T c = a;
    const T d = -b;

    const T score = a*b + c*d;
    flit::info_stream << id << ": score  = " << score  << std::endl;
    return score;
  }

protected:
  using flit::TestBase<T>::id;
};


REGISTER_TYPE(FMACancel)
