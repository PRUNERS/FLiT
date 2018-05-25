#ifndef SYMM_BASE_H
#define SYMM_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class SymmBase : public flit::TestBase<T> {
public:
  SymmBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*M*N + M*M; }
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
    std::vector<int> sizes = {M*N, M*M, M*N};
    std::vector<T> C = split_vector(sizes, 0, ti);
    std::vector<T> A = split_vector(sizes, 1, ti);
    std::vector<T> B = split_vector(sizes, 2, ti);

    int i, j, k;
    T temp2;

    for (i = 0; i < M; i++)
      for (j = 0; j < N; j++ )
	{
	  temp2 = 0;
	  for (k = 0; k < i; k++) {
	    C[k*M +j] += alpha*B[i*M +j] * A[i*M +k];
	    temp2 += B[k*M +j] * A[i*M +k];
	  }
	  C[i*M +j] = beta * C[i*M +j] + alpha*B[i*M +j] * A[i*M +i] + alpha * temp2;
	}

    return pickles({C});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // SYMM_BASE_H
