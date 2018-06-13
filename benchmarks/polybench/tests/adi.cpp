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
    auto u = ti;
    std::vector<T> v(N*N), p(N*N), q(N*N);

    int t, i, j;
    T DX, DY, DT;
    T B1, B2;
    T mul1, mul2;
    T a, b, c, d, e, f;

    DX = T(1.0) / T(N);
    DY = T(1.0) / T(N);
    DT = T(1.0) / T(TSTEPS);
    B1 = T(2.0);
    B2 = T(1.0);
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 / T(2.0);
    b = T(1.0) + mul1;
    c = a;
    d = -mul2 / T(2.0);
    e = T(1.0) + mul2;
    f = d;

    for (t = 1; t <= TSTEPS; t++) {
      // Column Sweep
      for (i = 1; i < N-1; i++) {
        v[0*N + i] = T(1.0);
        p[i*N + 0] = T(0.0);
        q[i*N + 0] = v[0*N + i];
        for (j = 1; j < N-1; j++) {
          p[i*N + j] = -c / (a*p[i*N + j-1]+b);
          q[i*N + j] =
            (-d*u[j*N + i-1] + (T(1.0) + T(2.0)*d) * u[j*N + i] - f*u[j*N + i+1]
             - a * q[i*N + j-1])
            / (a * p[i*N + j - 1] + b);
        }

        v[(N-1)*N + i] = T(1.0);
        for (j = N-2; j >= 1; j--) {
          v[j*N + i] = p[i*N + j] * v[(j+1)*N + i] + q[i*N + j];
        }
      }
      // Row Sweep
      for (i = 1; i < N-1; i++) {
        u[i*N + 0] = T(1.0);
        p[i*N + 0] = T(0.0);
        q[i*N + 0] = u[i*N + 0];
        for (j = 1; j < N-1; j++) {
          p[i*N + j] = -f / (d*p[i*N + j-1]+e);
          q[i*N + j] =
            (-a*v[(i-1)*N + j] + (T(1.0) + T(2.0)*a) * v[i*N + j]
             - c*v[(i+1)*N + j] - d*q[i*N + j-1])
            / (d * p[i*N + j-1] + e);
        }
        u[i*N + N-1] = T(1.0);
        for (j = N-2; j >= 1; j--) {
          u[i*N + j] = p[i*N + j] * u[i*N + j+1] + q[i*N + j];
        }
      }
    }

    return pickles({u, v, p, q});
  }

protected:
  using flit::TestBase<T>::id;
};

POLY_REGISTER_DIM2(Adi, 4, 4)
POLY_REGISTER_DIM2(Adi, 5, 5)
POLY_REGISTER_DIM2(Adi, 6, 6)
POLY_REGISTER_DIM2(Adi, 7, 7)
POLY_REGISTER_DIM2(Adi, 8, 8)
