#ifndef DURBIN_BASE_H
#define DURBIN_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class DurbinBase : public flit::TestBase<T> {
public:
  DurbinBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N};
    std::vector<T> r = split_vector(sizes, 0, ti);
    std::vector<T> y(N);

    std::vector<T> z(N);
    T alpha;
    T beta;
    T sum;

    int i,k;

    y[0] = -r[0];
    beta = static_cast<T>(1.0);
    alpha = -r[0];

    for (k = 1; k < N; k++) {
      beta = (1-alpha*alpha)*beta;
      sum = static_cast<T>(0.0);
      for (i=0; i<k; i++) {
	sum += r[k-i-1]*y[i];
      }
      alpha = - (r[k] + sum)/beta;

      for (i=0; i<k; i++) {
	z[i] = y[i] + alpha*y[k-i-1];
      }
      for (i=0; i<k; i++) {
	y[i] = z[i];
      }
      y[k] = alpha;
    }

    return pickles({y, z});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // DURBIN_BASE_H
