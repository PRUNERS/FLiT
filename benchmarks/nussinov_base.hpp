#ifndef NUSSINOV_BASE_H
#define NUSSINOV_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

template <typename T, int N>
class NussinovBase : public flit::TestBase<T> {
public:
  NussinovBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N + N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N, N*N};
    std::vector<T> seq = split_vector(sizes, 0, ti);
    std::vector<T> table = split_vector(sizes, 1, ti);

    int i, j, k;

    for (i = N-1; i >= 0; i--) {
      for (j=i+1; j<N; j++) {

	if (j-1>=0)
	  table[i*N + j] = max_score(table[i*N + j], table[i*N + j-1]);
	if (i+1<N)
	  table[i*N + j] = max_score(table[i*N + j], table[(i+1)*N + j]);

	if (j-1>=0 && i+1<N) {
	  /* don't allow adjacent elements to bond */
	  if (i<j-1)
	    table[i*N + j] = max_score(table[i*N + j], table[(i+1)*N + j-1]+match(seq[i], seq[j]));
	  else
	    table[i*N + j] = max_score(table[i*N + j], table[(i+1)*N + j-1]);
	}

	for (k=i+1; k<j; k++) {
	  table[i*N + j] = max_score(table[i*N + j], table[i*N + k] + table[(k+1)*N + j]);
	}
      }
    }

    return pickles({table});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // NUSSINOV_BASE_H
