#ifndef SEIDEL_2D_BASE_H
#define SEIDEL_2D_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int TSTEPS, int N>
class Seidel_2dBase : public flit::TestBase<T> {
public:
  Seidel_2dBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

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

    int t, i, j;

    for (t = 0; t <= TSTEPS - 1; t++)
      for (i = 1; i<= N - 2; i++)
	for (j = 1; j <= N - 2; j++)
	  A[(i)*N + (j)] = (A[(i-1)*N + (j-1)] + A[(i-1)*N + (j)] + A[(i-1)*N + (j+1)]
			    + A[(i)*N + (j-1)] + A[(i)*N + (j)] + A[(i)*N + (j+1)]
			    + A[(i+1)*N + (j-1)] + A[(i+1)*N + (j)] + A[(i+1)*N + (j+1)])/static_cast<T>(9.0);

    return pickles({A});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // SEIDEL_2D_BASE_H
