/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

#include "polybench_utils.h"

#include <flit.h>

#include <string>

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
    auto split = split_vector(ti, {W*H, W*H});
    auto &imgIn = split[0];
    auto &imgOut = split[1];
    std::vector<T> y1(W*H), y2(W*H);

    T alpha{0.25};

    int i,j;
    T xm1, tm1, ym1, ym2;
    T xp1, xp2;
    T tp1, tp2;
    T yp1, yp2;

    T k;
    T a1, a2, a3, a4, a5, a6, a7, a8;
    T b1, b2, c1, c2;

    k = (T(1.0) - std::exp(-alpha)) *
        (T(1.0) - std::exp(-alpha))
          / (T(1.0) + T(2.0) * alpha * std::exp(-alpha)
             - std::exp(T(2.0) * alpha));
    a1 = a5 = k;
    a2 = a6 = k * std::exp(-alpha) * (alpha - T(1.0));
    a3 = a7 = k * std::exp(-alpha) * (alpha + T(1.0));
    a4 = a8 = -k * std::exp(T(-2.0) * alpha);
    b1 =  std::pow(T(2.0), -alpha);
    b2 = -std::exp(T(-2.0) * alpha);
    c1 = c2 = 1;

    for (i = 0; i < W; i++) {
      ym1 = T(0.0);
      ym2 = T(0.0);
      xm1 = T(0.0);
      for (j = 0; j < H; j++) {
        y1[i*W + j] = a1 * imgIn[i*W + j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
        xm1 = imgIn[i*W + j];
        ym2 = ym1;
        ym1 = y1[i*W + j];
      }
    }

    for (i = 0; i < W; i++) {
      yp1 = T(0.0);
      yp2 = T(0.0);
      xp1 = T(0.0);
      xp2 = T(0.0);
      for (j = H-1; j >= 0; j--) {
        y2[i*W + j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
        xp2 = xp1;
        xp1 = imgIn[i*W + j];
        yp2 = yp1;
        yp1 = y2[i*W + j];
      }
    }

    for (i = 0; i < W; i++)
      for (j = 0; j < H; j++) {
        imgOut[i*W + j] = c1 * (y1[i*W + j] + y2[i*W + j]);
      }

    for (j = 0; j < H; j++) {
      tm1 = T(0.0);
      ym1 = T(0.0);
      ym2 = T(0.0);
      for (i = 0; i < W; i++) {
        y1[i*W + j] = a5 * imgOut[i*W + j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
        tm1 = imgOut[i*W + j];
        ym2 = ym1;
        ym1 = y1[i*W + j];
      }
    }


    for (j = 0; j < H; j++) {
      tp1 = T(0.0);
      tp2 = T(0.0);
      yp1 = T(0.0);
      yp2 = T(0.0);
      for (i = W-1; i >= 0; i--) {
        y2[i*W + j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
        tp2 = tp1;
        tp1 = imgOut[i*W + j];
        yp2 = yp1;
        yp1 = y2[i*W + j];
      }
    }

    for (i = 0; i < W; i++) {
      for (j = 0; j < H; j++) {
        imgOut[i*W + j] = c2 * (y1[i*W + j] + y2[i*W + j]);
      }
    }

    return pickles({imgOut, y1, y2});
  }

protected:
  using flit::TestBase<T>::id;
};

POLY_REGISTER_DIM2(Deriche, 4, 4)
POLY_REGISTER_DIM2(Deriche, 5, 5)
POLY_REGISTER_DIM2(Deriche, 6, 6)
POLY_REGISTER_DIM2(Deriche, 7, 7)
POLY_REGISTER_DIM2(Deriche, 8, 8)
