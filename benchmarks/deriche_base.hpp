#ifndef DERICHE_BASE_H
#define DERICHE_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int W, int H>
class DericheBase : public flit::TestBase<T> {
public:
  DericheBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 2*W*H; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {W*H, W*H};
    std::vector<T> imgIn = split_vector(sizes, 0, ti);
    std::vector<T> imgOut = split_vector(sizes, 1, ti);
    std::vector<T> y1(W*H);
    std::vector<T> y2(W*H);

    T alpha = static_cast<T>(0.25);

    int i,j;
    T xm1, tm1, ym1, ym2;
    T xp1, xp2;
    T tp1, tp2;
    T yp1, yp2;

    T k;
    T a1, a2, a3, a4, a5, a6, a7, a8;
    T b1, b2, c1, c2;

    k = (static_cast<T>(1.0)-std::exp(-alpha))*(static_cast<T>(1.0)-std::exp(-alpha))/(static_cast<T>(1.0)+static_cast<T>(2.0)*alpha*std::exp(-alpha)-std::exp(static_cast<T>(2.0)*alpha));
    a1 = a5 = k;
    a2 = a6 = k*std::exp(-alpha)*(alpha-static_cast<T>(1.0));
    a3 = a7 = k*std::exp(-alpha)*(alpha+static_cast<T>(1.0));
    a4 = a8 = -k*std::exp(static_cast<T>(-2.0)*alpha);
    b1 =  std::pow(static_cast<T>(2.0),-alpha);
    b2 = -std::exp(static_cast<T>(-2.0)*alpha);
    c1 = c2 = 1;

    for (i=0; i<W; i++) {
      ym1 = static_cast<T>(0.0);
      ym2 = static_cast<T>(0.0);
      xm1 = static_cast<T>(0.0);
      for (j=0; j<H; j++) {
	y1[i*W + j] = a1*imgIn[i*W + j] + a2*xm1 + b1*ym1 + b2*ym2;
	xm1 = imgIn[i*W + j];
	ym2 = ym1;
	ym1 = y1[i*W + j];
      }
    }

    for (i=0; i<W; i++) {
      yp1 = static_cast<T>(0.0);
      yp2 = static_cast<T>(0.0);
      xp1 = static_cast<T>(0.0);
      xp2 = static_cast<T>(0.0);
      for (j=H-1; j>=0; j--) {
	y2[i*W + j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
	xp2 = xp1;
	xp1 = imgIn[i*W + j];
	yp2 = yp1;
	yp1 = y2[i*W + j];
      }
    }

    for (i=0; i<W; i++)
      for (j=0; j<H; j++) {
	imgOut[i*W + j] = c1 * (y1[i*W + j] + y2[i*W + j]);
      }

    for (j=0; j<H; j++) {
      tm1 = static_cast<T>(0.0);
      ym1 = static_cast<T>(0.0);
      ym2 = static_cast<T>(0.0);
      for (i=0; i<W; i++) {
	y1[i*W + j] = a5*imgOut[i*W + j] + a6*tm1 + b1*ym1 + b2*ym2;
	tm1 = imgOut[i*W + j];
	ym2 = ym1;
	ym1 = y1 [i*W + j];
      }
    }


    for (j=0; j<H; j++) {
      tp1 = static_cast<T>(0.0);
      tp2 = static_cast<T>(0.0);
      yp1 = static_cast<T>(0.0);
      yp2 = static_cast<T>(0.0);
      for (i=W-1; i>=0; i--) {
	y2[i*W + j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
	tp2 = tp1;
	tp1 = imgOut[i*W + j];
	yp2 = yp1;
	yp1 = y2[i*W + j];
      }
    }

    for (i=0; i<W; i++)
      for (j=0; j<H; j++)
	imgOut[i*W + j] = c2*(y1[i*W + j] + y2[i*W + j]);

    return pickles({imgOut, y1, y2});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // DERICHE_BASE_H
