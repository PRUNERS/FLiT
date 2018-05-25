#ifndef GEMVER_BASE_H
#define GEMVER_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class GemverBase : public flit::TestBase<T> {
public:
  GemverBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*N + 8*N; }
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
    std::vector<int> sizes = {N*N, N, N, N, N, N, N, N, N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> u1 = split_vector(sizes, 1, ti);
    std::vector<T> v1 = split_vector(sizes, 2, ti);
    std::vector<T> u2 = split_vector(sizes, 3, ti);
    std::vector<T> v2 = split_vector(sizes, 4, ti);
    std::vector<T> w = split_vector(sizes, 5, ti);
    std::vector<T> x = split_vector(sizes, 6, ti);
    std::vector<T> y = split_vector(sizes, 7, ti);
    std::vector<T> z = split_vector(sizes, 8, ti);

    int i,j;

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
	A[i*N + j] = A[i*N + j] + u1[i] * v1[j] + u2[i] * v2[j];
      }
    }

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
	x[i] = x[i] + beta * A[j*N + i] * y[j];
      }
    }

    for (i = 0; i < N; i++) {
      x[i] = x[i] + z[i];
    }

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
	w[i] = w[i] +  alpha * A[i*N + j] * x[j];
      }
    }

    return pickles({A, w, x});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // GEMVER_BASE_H
