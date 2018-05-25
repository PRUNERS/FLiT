#ifndef HEAT_3D_BASE_H
#define HEAT_3D_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int TSTEPS, int N>
class Heat_3dBase : public flit::TestBase<T> {
public:
  Heat_3dBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*N*N*N; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {N*N*N, N*N*N};
    std::vector<T> A = split_vector(sizes, 0, ti);
    std::vector<T> B = split_vector(sizes, 1, ti);

    int t, i, j, k;

    for (t = 1; t <= TSTEPS; t++) {
      for (i = 1; i < N-1; i++) {
	for (j = 1; j < N-1; j++) {
	  for (k = 1; k < N-1; k++) {
	    B[i*N*N + j*N +k] =   static_cast<T>(0.125) * (A[(i+1)*N*N + j*N +k] - static_cast<T>(2.0) * A[i*N*N + j*N +k] + A[(i-1)*N*N + j*N +k])
	      + static_cast<T>(0.125) * (A[i*N*N + (j+1)*N +k] - static_cast<T>(2.0) * A[i*N*N + j*N +k] + A[i*N*N + (j-1)*N*N + k])
	      + static_cast<T>(0.125) * (A[i*N*N + j*N +k+1] - static_cast<T>(2.0) * A[i*N*N + j*N +k] + A[i*N*N + j*N +k-1])
	      + A[i*N*N + j*N +k];
	  }
	}
      }
      for (i = 1; i < N-1; i++) {
	for (j = 1; j < N-1; j++) {
	  for (k = 1; k < N-1; k++) {
	    A[i*N*N + j*N +k] =   static_cast<T>(0.125) * (B[(i+1)*N*N + j*N +k] - static_cast<T>(2.0) * B[i*N*N + j*N +k] + B[(i-1)*N*N + j*N +k])
	      + static_cast<T>(0.125) * (B[i*N*N + (j+1)*N*N + k] - static_cast<T>(2.0) * B[i*N*N + j*N +k] + B[i*N*N + (j-1)*N*N + k])
	      + static_cast<T>(0.125) * (B[i*N*N + j*N +k+1] - static_cast<T>(2.0) * B[i*N*N + j*N +k] + B[i*N*N + j*N +k-1])
	      + B[i*N*N + j*N +k];
	  }
	}
      }
    }

    return pickles({A, B});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // HEAT_3D_BASE_H
