#ifndef FDTD_2D_BASE_H
#define FDTD_2D_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int TMAX, int NX, int NY>
class Fdtd_2dBase : public flit::TestBase<T> {
public:
  Fdtd_2dBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 3*NX*NY + TMAX; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {NX*NY, NX*NY, NX*NY, TMAX};
    std::vector<T> ex = split_vector(sizes, 0, ti);
    std::vector<T> ey = split_vector(sizes, 1, ti);
    std::vector<T> hz = split_vector(sizes, 2, ti);
    std::vector<T> _fict_ = split_vector(sizes, 3, ti);

    int t, i, j;

    for(t = 0; t < TMAX; t++)
      {
	for (j = 0; j < NY; j++)
	  ey[0*NX + j] = _fict_[t];
	for (i = 1; i < NX; i++)
	  for (j = 0; j < NY; j++)
	    ey[i*NX + j] = ey[i*NX + j] - static_cast<T>(0.5)*(hz[i*NX + j]-hz[(i-1)*NX + j]);
	for (i = 0; i < NX; i++)
	  for (j = 1; j < NY; j++)
	    ex[i*NX + j] = ex[i*NX + j] - static_cast<T>(0.5)*(hz[i*NX + j]-hz[i*NX + j-1]);
	for (i = 0; i < NX - 1; i++)
	  for (j = 0; j < NY - 1; j++)
	    hz[i*NX + j] = hz[i*NX + j] - static_cast<T>(0.7)*  (ex[i*NX + j+1] - ex[i*NX + j] +
								 ey[(i+1)*NX + j] - ey[i*NX + j]);
      }

    return pickles({ex, ey, hz});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // FDTD_2D_BASE_H
