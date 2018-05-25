#ifndef MVT_BASE_H
#define MVT_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class MvtBase : public flit::TestBase<T> {
public:
  MvtBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 4*N + N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N, N, N, N, N*N};
    std::vector<T> x1 = split_vector(sizes, 0, ti);
    std::vector<T> x2 = split_vector(sizes, 1, ti);
    std::vector<T> y_1 = split_vector(sizes, 2, ti);
    std::vector<T> y_2 = split_vector(sizes, 3, ti);
    std::vector<T> A = split_vector(sizes, 4, ti);

    int i, j;

    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
	x1[i] = x1[i] + A[i*N +j] * y_1[j];
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
	x2[i] = x2[i] + A[j*N + i] * y_2[j];

    return pickles({x1, x2});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // MVT_BASE_H
