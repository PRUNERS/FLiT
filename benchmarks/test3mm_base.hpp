#ifndef TEST3MM_BASE_H
#define TEST3MM_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int NI, int NJ, int NK, int NL, int NM>
class Test3mmBase : public flit::TestBase<T> {
public:
  Test3mmBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return NI*NK + NK*NJ + NJ*NM + NM*NL; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {NI*NK, NK*NJ, NJ*NM, NM*NL};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);
    std::vector<T> C = split_vector(sizes, 2, ti);
    std::vector<T> D = split_vector(sizes, 3, ti);
    std::vector<T> E(NI*NJ);
    std::vector<T> F(NJ*NL);
    std::vector<T> G(NI*NL);

    int i, j, k;

    /* E := A*B */
    for (i = 0; i < NI; i++)
      for (j = 0; j < NJ; j++)
	{
	  E[i*NI + j] = static_cast<T>(0.0);
	  for (k = 0; k < NK; ++k)
	    E[i*NI + j] += A[i*NI + k] * B[k*NK + j];
	}
    /* F := C*D */
    for (i = 0; i < NJ; i++)
      for (j = 0; j < NL; j++)
	{
	  F[i*NJ + j] = static_cast<T>(0.0);
	  for (k = 0; k < NM; ++k)
	    F[i*NJ + j] += C[i*NJ + k] * D[k*NM + j];
	}
    /* G := E*F */
    for (i = 0; i < NI; i++)
      for (j = 0; j < NL; j++)
	{
	  G[i*NI + j] = static_cast<T>(0.0);
	  for (k = 0; k < NJ; ++k)
	    G[i*NI + j] += E[i*NI + k] * F[k*NJ + j];
	}

    return pickles({E, F, G});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // TEST3MM_BASE_H
