#ifndef TRISOLV_BASE_H
#define TRISOLV_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class TrisolvBase : public flit::TestBase<T> {
public:
  TrisolvBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*N + 2*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N, N, N};
    std::vector<T> L = split_vector(sizes, 0, ti);
    std::vector<T> x = split_vector(sizes, 1, ti);
    std::vector<T> b = split_vector(sizes, 2, ti);

    int i, j;

    for (i = 0; i < N; i++)
      {
	x[i] = b[i];
	for (j = 0; j <i; j++)
	  x[i] -= L[i*N + j] * x[j];
	x[i] = x[i] / L[i*N + i];
      }

    return pickles({x});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // TRISOLV_BASE_H
