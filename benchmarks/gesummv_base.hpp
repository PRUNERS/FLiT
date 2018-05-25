#ifndef GESUMMV_BASE_H
#define GESUMMV_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class GesummvBase : public flit::TestBase<T> {
public:
  GesummvBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*N*N + N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    T alpha = static_cast<T>(1.5);
    T beta = static_cast<T>(1.2);
    std::vector<int> sizes = {N*N, N*N, N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);
    std::vector<T> x = split_vector(sizes, 2, ti);

    std::vector<T> tmp(N);
    std::vector<T> y(N);

    int i, j;

    for (i = 0; i < N; i++)
      {
	tmp[i] = static_cast<T>(0.0);
	y[i] = static_cast<T>(0.0);
	for (j = 0; j < N; j++)
	  {
	    tmp[i] = A[i*N + j] * x[j] + tmp[i];
	    y[i] = B[i*N + j] * x[j] + y[i];
	  }
	y[i] = alpha * tmp[i] + beta * y[i];
      }

    return pickles({tmp, y});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // GESUMMV_BASE_H
