#ifndef GEMM_BASE_H
#define GEMM_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int NI, int NJ, int NK>
class GemmBase : public flit::TestBase<T> {
public:
  GemmBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return NI*NJ + NI*NK + NK+NJ; }
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
    std::vector<int> sizes = {NI*NI, NI*NK, NK+NJ};
    std::vector<T> C = split_vector(sizes, 0, ti);
    std::vector<T> A = split_vector(sizes, 1, ti);
    std::vector<T> B = split_vector(sizes, 2, ti);

    int i, j, k;

    for (i = 0; i < NI; i++) {
      for (j = 0; j < NJ; j++) {
	C[i*NI + j] *= beta;
      }
      for (k = 0; k < NK; k++) {
	for (j = 0; j < NJ; j++) {
	  C[i*NI + j] += alpha * A[i*NI + k] * B[k*NK + j];
	}
      }
    }

    return pickles({C});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // GEMM_BASE_H
