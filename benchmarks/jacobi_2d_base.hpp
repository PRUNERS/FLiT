#ifndef JACOBI_2D_BASE_H
#define JACOBI_2D_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int TSTEPS, int N>
class Jacobi_2dBase : public flit::TestBase<T> {
public:
  Jacobi_2dBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N, N*N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);

    int t, i, j;

    for (t = 0; t < TSTEPS; t++)
      {
	for (i = 1; i < N - 1; i++)
	  for (j = 1; j < N - 1; j++)
	    B[i*N + j] = static_cast<T>(0.2) * (A[i*N + j] + A[i*N + j-1] + A[i*N + 1+j] + A[(1+i)*N + j] + A[(i-1)*N + j]);
	for (i = 1; i < N - 1; i++)
	  for (j = 1; j < N - 1; j++)
	    A[i*N + j] = static_cast<T>(0.2) * (B[i*N + j] + B[i*N + j-1] + B[i*N + 1+j] + B[(1+i)*N + j] + B[(i-1)*N + j]);
      }

    return pickles({A, B});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // JACOBI_2D_BASE_H
