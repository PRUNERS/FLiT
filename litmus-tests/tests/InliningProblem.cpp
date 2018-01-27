#include <flit.h>

#include <vector>

#include <cstdlib>

template <typename T>
class InliningProblem : public flit::TestBase<T> {
public:
  InliningProblem(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 1; }

  virtual std::vector<T> getDefaultInput() override {
    return { .1, 1.1e3, -.1, -1.1e3, 1/3 };
  }

protected:
  T identity(const T x) {
    const T nx = -x;
    const T x_again = -nx;
    return x_again;
  }
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T a = ti[0];
    T also_a = identity(a);

    const T score = std::sqrt(a) * std::sqrt(also_a);

    flit::info_stream << id << ": score  = " << score  << std::endl;

    return score;
  }

protected:
  using flit::TestBase<T>::id;
};


REGISTER_TYPE(InliningProblem)
