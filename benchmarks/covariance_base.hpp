#ifndef COVARIANCE_BASE_H
#define COVARIANCE_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class CovarianceBase : public flit::TestBase<T> {
public:
  CovarianceBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*M; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    T float_n = static_cast<T>(N);
    std::vector<T> data = ti;
    std::vector<T> cov(M*M);
    std::vector<T> mean(M);

    int i, j, k;

    for (j = 0; j < M; j++) {
      mean[j] = static_cast<T>(0.0);
      for (i = 0; i < N; i++) {
        mean[j] += data[i*N + j];
      }
      mean[j] /= float_n;
    }

    for (i = 0; i < N; i++) {
      for (j = 0; j < M; j++) {
	data[i*N + j] -= mean[j];
      }
    }

    for (i = 0; i < M; i++) {
      for (j = i; j < M; j++) {
        cov[i*M + j] = static_cast<T>(0.0);
        for (k = 0; k < N; k++) {
	  cov[i*M + j] += data[k*N + i] * data[k*N + j];
	}
        cov[i*M + j] /= (float_n - static_cast<T>(1.0));
        cov[j*M + i] = cov[i*M + j];
      }
    }

    return pickles({data, cov, mean}) ;
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // COVARIANCE_BASE_H
