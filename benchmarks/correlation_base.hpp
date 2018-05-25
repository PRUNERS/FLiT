#ifndef CORRELATION_BASE_H
#define CORRELATION_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class CorrelationBase : public flit::TestBase<T> {
public:
  CorrelationBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

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
    std::vector<T> data = ti;
    std::vector<T> corr (M*M);
    std::vector<T> mean (M);
    std::vector<T> stddev (M);
    T float_n = N;

    int i, j, k;

    T eps = 0.1;


    for (j=0; j<M; j++) {
      mean[j] = 0.0;
      for (i=0; i<N; i++) {
	mean[j] += data[i*N + j];
      }
      mean[j] /= float_n;
    }


    for (j=0; j<M; j++) {
      stddev[j] = 0.0;
      for (i=0; i<N; i++) {
        stddev[j] += (data[i*N + j] - mean[j]) * (data[i*N + j] - mean[j]);
      }
      stddev[j] /= float_n;
      stddev[j] = std::sqrt(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

    /* Center and reduce the column vectors. */
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        data[i*N + j] -= mean[j];
        data[i*N + j] /= std::sqrt(float_n) * stddev[j];
      }
    }

    /* Calculate the m * m correlation matrix. */
    for (i=0; i<M-1; i++) {
      corr[i*M + i] = 1.0;
      for (j=i+1; j<M; j++) {
	corr[i*M + j] = 0.0;
	for (k=0; k<N; k++) {
	  corr[i*M + j] += (data[k*N + i] * data[k*N + j]);
	}
	corr[j*M + i] = corr[i*M + j];
      }
    }
    corr[(M-1)*M + (M-1)] = 1.0;


    return pickles({data, corr, mean, stddev}) ;
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // CORRELATION_BASE_H
