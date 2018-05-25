#ifndef SYR2K_BASE_H
#define SYR2K_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class Syr2kBase : public flit::TestBase<T> {
public:
  Syr2kBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*N + 2*N*M; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N, N*M, N*M};
    std::vector<T> C = split_vector(sizes, 0, ti);
    std::vector<T> A = split_vector(sizes, 1, ti);
    std::vector<T> B = split_vector(sizes, 2, ti);
    T alpha = static_cast<T>(1.5);
    T beta = static_cast<T>(1.2);

    int i, j, k;

    for (i = 0; i < N; i++) {
      for (j = 0; j <= i; j++)
	C[i*N + j] *= beta;
      for (k = 0; k < M; k++)
	for (j = 0; j <= i; j++)
	  {
	    C[i*N + j] += A[j*N + k]*alpha*B[i*N + k] + B[j*N + k]*alpha*A[i*N + k];
	  }
    }


    return pickles({C});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // SYR2K_BASE_H
