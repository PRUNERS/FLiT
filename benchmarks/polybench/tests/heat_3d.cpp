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
    auto split = split_vector(ti, {N*N*N, N*N*N});
    auto &A = split[0];
    auto &B = split[1];

    int t, i, j, k;

    for (t = 1; t <= TSTEPS; t++) {
      for (i = 1; i < N-1; i++) {
        for (j = 1; j < N-1; j++) {
          for (k = 1; k < N-1; k++) {
            B[i*N*N + j*N +k] =
                T(0.125) * (A[(i+1)*N*N + j*N +k] - T(2.0) * A[i*N*N + j*N +k] +
                            A[(i-1)*N*N + j*N +k])
                +
                T(0.125) * (A[i*N*N + (j+1)*N +k] - T(2.0) * A[i*N*N + j*N +k] +
                            A[i*N*N + (j-1)*N*N + k])
                +
                T(0.125) * (A[i*N*N + j*N +k+1] - T(2.0) * A[i*N*N + j*N +k] +
                            A[i*N*N + j*N +k-1])
                +
                A[i*N*N + j*N +k];
          }
        }
      }
      for (i = 1; i < N-1; i++) {
        for (j = 1; j < N-1; j++) {
          for (k = 1; k < N-1; k++) {
            A[i*N*N + j*N +k] =
                T(0.125) * (B[(i+1)*N*N + j*N +k] - T(2.0) * B[i*N*N + j*N +k] +
                            B[(i-1)*N*N + j*N +k])
                +
                T(0.125) * (B[i*N*N + (j+1)*N*N + k] -
                            T(2.0) * B[i*N*N + j*N +k] +
                            B[i*N*N + (j-1)*N*N + k])
                +
                T(0.125) * (B[i*N*N + j*N + k+1] - T(2.0) * B[i*N*N + j*N + k] +
                            B[i*N*N + j*N + k-1])
                +
                B[i*N*N + j*N +k];
          }
        }
      }
    }

    return pickles({A, B});
  }

protected:
  using flit::TestBase<T>::id;
};

POLY_REGISTER_DIM2(Heat_3d, 4, 4)
POLY_REGISTER_DIM2(Heat_3d, 5, 5)
POLY_REGISTER_DIM2(Heat_3d, 6, 6)
POLY_REGISTER_DIM2(Heat_3d, 7, 7)
POLY_REGISTER_DIM2(Heat_3d, 8, 8)
