#ifndef DOITGEN_BASE_H
#define DOITGEN_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int NR, int NQ, int NP>
class DoitgenBase : public flit::TestBase<T> {
public:
  DoitgenBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return NR*NQ*NP + NP*NP; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {NR*NQ*NP, NP*NP};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> C4 = split_vector(sizes, 1, ti);
    std::vector<T> sum(NP);

    int r, q, p, s;

    for (r = 0; r < NR; r++)
      for (q = 0; q < NQ; q++)  {
	for (p = 0; p < NP; p++)  {
	  sum[p] = static_cast<T>(0.0);
	  for (s = 0; s < NP; s++)
	    sum[p] += A[r*NR*NQ + q*NQ + s] * C4[s*NP + p];
	}
	for (p = 0; p < NP; p++)
	  A[r*NR*NQ + q*NQ + p] = sum[p];
      }


    return pickles({A, sum});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // DOITGEN_BASE_H
