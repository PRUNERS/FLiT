#include <flit.h>

#include <vector>

#include <cstdlib>

template <typename T>
class FMACancel : public flit::TestBase<T> {
public:
  FMACancel(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2; }

  virtual std::vector<T> getDefaultInput() override {
    return { .1, 1.1e5 };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    const T a = ti[0];
    const T b = ti[1];
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
