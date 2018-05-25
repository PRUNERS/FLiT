#ifndef LUDCMP_BASE_H
#define LUDCMP_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class LudcmpBase : public flit::TestBase<T> {
public:
  LudcmpBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*N + 3*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N, N, N, N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> b = split_vector(sizes, 1, ti);
    std::vector<T> x = split_vector(sizes, 2, ti);
    std::vector<T> y = split_vector(sizes, 3, ti);

    int i, j, k;

    T w;

    for (i = 0; i < N; i++) {
      for (j = 0; j <i; j++) {
	w = A[i*N + j];
	for (k = 0; k < j; k++) {
          w -= A[i*N + k] * A[k*N + j];
	}
        A[i*N + j] = w / A[j*N + j];
      }
      for (j = i; j < N; j++) {
	w = A[i*N + j];
	for (k = 0; k < i; k++) {
          w -= A[i*N + k] * A[k*N + j];
	}
	A[i*N + j] = w;
      }
    }

    for (i = 0; i < N; i++) {
      w = b[i];
      for (j = 0; j < i; j++)
        w -= A[i*N + j] * y[j];
      y[i] = w;
    }

    for (i = N-1; i >=0; i--) {
      w = y[i];
      for (j = i+1; j < N; j++)
        w -= A[i*N + j] * x[j];
      x[i] = w / A[i*N + i];
    }

    return pickles({A, x, y});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // LUDCMP_BASE_H
