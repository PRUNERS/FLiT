#ifndef GRAMSCHMIDT_BASE_H
#define GRAMSCHMIDT_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class GramschmidtBase : public flit::TestBase<T> {
public:
  GramschmidtBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*M*N + N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {M*N, N*N, M*N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> R = split_vector(sizes, 1, ti);
    std::vector<T> Q = split_vector(sizes, 2, ti);

    int i, j, k;

    T nrm;

    for (k = 0; k < N; k++)
      {
	nrm = static_cast<T>(0.0);
	for (i = 0; i < M; i++)
	  nrm += A[i*M + k] * A[i*M + k];
	R[k*N + k] = std::sqrt(nrm);
	for (i = 0; i < M; i++)
	  Q[i*M + k] = A[i*M + k] / R[k*N + k];
	for (j = k + 1; j < N; j++)
	  {
	    R[k*N + j] = static_cast<T>(0.0);
	    for (i = 0; i < M; i++)
	      R[k*N + j] += Q[i*M + k] * A[i*M + j];
	    for (i = 0; i < M; i++)
	      A[i*M + j] = A[i*M + j] - Q[i*M + k] * R[k*N + j];
	  }
      }

    return pickles({A, R, Q});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // GRAMSCHMIDT_BASE_H
