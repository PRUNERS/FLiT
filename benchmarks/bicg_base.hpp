#ifndef BICG_BASE_H
#define BICG_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class BicgBase : public flit::TestBase<T> {
public:
  BicgBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*M + N + M; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*M, N, M};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> r = split_vector(sizes, 1, ti);
    std::vector<T> p = split_vector(sizes, 2, ti);
    std::vector<T> s(M);
    std::vector<T> q(N);

    int i, j;

    for (i = 0; i < M; i++)
      s[i] = 0;
    for (i = 0; i < N; i++)
      {
	q[i] = static_cast<T>(0.0);
	for (j = 0; j < M; j++)
	  {
	    s[j] = s[j] + r[i] * A[i*N + j];
	    q[i] = q[i] + A[i*N + j] * p[j];
	  }
      }

    return pickles({s, q});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // BICG_BASE_H
