#ifndef TRMM_BASE_H
#define TRMM_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class TrmmBase : public flit::TestBase<T> {
public:
  TrmmBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return M*M + M*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {M*M, M*N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);
    T alpha = static_cast<T>(1.5);

    int i, j, k;

    for (i = 0; i < M; i++)
      for (j = 0; j < N; j++) {
        for (k = i+1; k < M; k++)
	  B[i*M + j] += A[k*M + i] * B[k*M + j];
        B[i*M + j] = alpha * B[i*M + j];
      }


    return pickles({B});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // TRMM_BASE_H
