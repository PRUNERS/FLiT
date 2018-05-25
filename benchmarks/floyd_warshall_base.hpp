#ifndef FLOYD_WARSHALL_BASE_H
#define FLOYD_WARSHALL_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N>
class Floyd_WarshallBase : public flit::TestBase<T> {
public:
  Floyd_WarshallBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

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
    std::vector<T> path = split_vector(sizes, 0, ti);

    int i, j, k;

    for (k = 0; k < N; k++)
      {
	for(i = 0; i < N; i++)
	  for (j = 0; j < N; j++)
	    path[i*N + j] = path[i*N + j] < path[i*N + k] + path[k*N + j] ?
					    path[i*N + j] : path[i*N + k] + path[k*N + j];
      }

    return pickles({path});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // FLOYD_WARSHALL_BASE_H
