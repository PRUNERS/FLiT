#ifndef TEST2MM_BASE_H
#define TEST2MM_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int NI, int NJ, int NK, int NL>
class Test2mmBase : public flit::TestBase<T> {
public:
  Test2mmBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return NI*NK + NK*NJ + NJ*NL + NI*NL; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {NI*NK, NK*NJ, NJ*NL, NI*NL};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);
    std::vector<T> C = split_vector(sizes, 2, ti);
    std::vector<T> D = split_vector(sizes, 3, ti);
    std::vector<T> tmp(NI*NJ);
    T alpha = static_cast<T>(1.5);
    T beta = static_cast<T>(1.2);

    int i, j, k;

    /* D := alpha*A*B*C + beta*D */
    for (i = 0; i < NI; i++)
      for (j = 0; j < NJ; j++)
	{
	  tmp[i*NI + j] = static_cast<T>(0.0);
	  for (k = 0; k < NK; ++k)
	    tmp[i*NI + j] += alpha * A[i*NI + k] * B[k*NK + j];
	}
    for (i = 0; i < NI; i++)
      for (j = 0; j < NL; j++)
	{
	  D[i*NI + j] *= beta;
	  for (k = 0; k < NJ; ++k)
	    D[i*NI + j] += tmp[i*NI + k] * C[k*NJ + j];
	}

    return pickles({tmp, D});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // TEST2MM_BASE_H
