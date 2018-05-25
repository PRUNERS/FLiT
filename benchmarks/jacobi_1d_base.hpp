#ifndef JACOBI_1D_BASE_H
#define JACOBI_1D_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int TSTEPS, int N>
class Jacobi_1dBase : public flit::TestBase<T> {
public:
  Jacobi_1dBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N, N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);

    int t, i;

    for (t = 0; t < TSTEPS; t++)
      {
	for (i = 1; i < N - 1; i++)
	  B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
	for (i = 1; i < N - 1; i++)
	  A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      }

    return pickles({A, B});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // JACOBI_1D_BASE_H
