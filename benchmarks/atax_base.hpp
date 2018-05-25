#ifndef ATAX_BASE_H
#define ATAX_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class AtaxBase : public flit::TestBase<T> {
public:
  AtaxBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return M*N + N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {M*N, N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> x = split_vector(sizes, 1, ti);
    std::vector<T> y(N);
    std::vector<T> tmp(M);

    int i,j;

    for (i = 0; i < N; i++)
      y[i] = 0;
    for (i = 0; i < M; i++)
      {
	tmp[i] = static_cast<T>(0.0);
	for (j = 0; j < N; j++)
	  tmp[i] = tmp[i] + A[i*M + j] * x[j];
	for (j = 0; j < N; j++)
	  y[j] = y[j] + A[i*M + j] * tmp[i];
      }


    return pickles({y, tmp});
}

protected:
using flit::TestBase<T>::id;
};

#endif  // ATAX_BASE_H
