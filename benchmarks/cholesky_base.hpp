#ifndef CHOLESKY_BASE_H
#define CHOLESKY_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class CholeskyBase : public flit::TestBase<T> {
public:
  CholeskyBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N};
    std::vector<T> A = split_vector(sizes, 0, ti);

    int i, j, k;

    for (i = 0; i < N; i++) {
      //j<i
      for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
	  A[i*N + j] -= A[i*N + k] * A[j*N + k];
        }
        A[i*N + j] /= A[j*N + j];
      }
      // i==j case
      for (k = 0; k < i; k++) {
        A[i*N + i] -= A[i*N + k] * A[i*N + k];
      }
      A[i*N + i] = std::sqrt(A[i*N + i]);
    }

    return pickles({A});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // CHOLESKY_BASE_H
