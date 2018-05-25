#ifndef LU_BASE_H
#define LU_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class LuBase : public flit::TestBase<T> {
public:
  LuBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

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
      for (j = 0; j <i; j++) {
	for (k = 0; k < j; k++) {
          A[i*N + j] -= A[i*N + k] * A[k*N + j];
	}
        A[i*N + j] /= A[j*N + j];
      }
      for (j = i; j < N; j++) {
	for (k = 0; k < i; k++) {
          A[i*N + j] -= A[i*N + k] * A[k*N + j];
	}
      }
    }

    return pickles({A});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // LU_BASE_H
