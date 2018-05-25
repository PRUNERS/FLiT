#ifndef ADI_BASE_H
#define ADI_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int N, int TSTEPS>
class AdiBase : public flit::TestBase<T> {
public:
  AdiBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

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
    std::vector<T> u = split_vector(sizes, 0, ti);
    std::vector<T> v(N*N);
    std::vector<T> p(N*N);
    std::vector<T> q(N*N);

    int t, i, j;
    T DX, DY, DT;
    T B1, B2;
    T mul1, mul2;
    T a, b, c, d, e, f;

    DX = static_cast<T>(1.0)/(T)N;
    DY = static_cast<T>(1.0)/(T)N;
    DT = static_cast<T>(1.0)/(T)TSTEPS;
    B1 = static_cast<T>(2.0);
    B2 = static_cast<T>(1.0);
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 /  static_cast<T>(2.0);
    b = static_cast<T>(1.0)+mul1;
    c = a;
    d = -mul2 / static_cast<T>(2.0);
    e = static_cast<T>(1.0)+mul2;
    f = d;

    for (t=1; t<=TSTEPS; t++) {
      //Column Sweep
      for (i=1; i<N-1; i++) {
	v[0*N + i] = static_cast<T>(1.0);
	p[i*N + 0] = static_cast<T>(0.0);
	q[i*N + 0] = v[0*N + i];
	for (j=1; j<N-1; j++) {
	  p[i*N + j] = -c / (a*p[i*N + j-1]+b);
	  q[i*N + j] = (-d*u[j*N + i-1]+(static_cast<T>(1.0)+static_cast<T>(2.0)*d)*u[j*N + i] - f*u[j*N + i+1]-a*q[i*N + j-1])/(a*p[i*N + j-1]+b);
	}

	v[(N-1)*N + i] = static_cast<T>(1.0);
	for (j=N-2; j>=1; j--) {
	  v[j*N + i] = p[i*N + j] * v[(j+1)*N + i] + q[i*N + j];
	}
      }
      //Row Sweep
      for (i=1; i<N-1; i++) {
	u[i*N + 0] = static_cast<T>(1.0);
	p[i*N + 0] = static_cast<T>(0.0);
	q[i*N + 0] = u[i*N + 0];
	for (j=1; j<N-1; j++) {
	  p[i*N + j] = -f / (d*p[i*N + j-1]+e);
	  q[i*N + j] = (-a*v[(i-1)*N + j]+(static_cast<T>(1.0)+static_cast<T>(2.0)*a)*v[i*N + j] - c*v[(i+1)*N + j]-d*q[i*N + j-1])/(d*p[i*N + j-1]+e);
	}
	u[i*N + N-1] = static_cast<T>(1.0);
	for (j=N-2; j>=1; j--) {
	  u[i*N + j] = p[i*N + j] * u[i*N + j+1] + q[i*N + j];
	}
      }
    }

    return pickles({u, v, p, q});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // ADI_BASE_H
