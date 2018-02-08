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

#ifndef MATRIX_CU_H
#define MATRIX_CU_H

#include "CUHelpers.h"
#include "VectorCU.h"
#include "cuvector.h"

namespace flit {

template<typename T>
class MatrixCU {
  using rdtype = cuvector<T>;
  cuvector<rdtype> data;
public:
  using vsize_t = typename cuvector<T>::cvs_t;

  HOST_DEVICE
  MatrixCU(vsize_t rows, vsize_t cols):
    data(rows, cuvector<T>(cols,0)){}

  HOST_DEVICE
  inline
  rdtype&
  operator[](vsize_t indx){
    return data[indx];
  }

  HOST_DEVICE
  inline
  rdtype
  operator[](vsize_t indx) const {
    return data[indx];
  }

  HOST_DEVICE
  bool
  operator==(MatrixCU<T> const &rhs) const {
    if(data.size() != rhs.data.size()) return false;
    bool retVal = true;
    for(vsize_t x = 0; x < data.size(); ++x){
      for(vsize_t y = 0; y < data[0].size(); ++y){
        if(data[x][y] != rhs.data[x][y]){
          retVal = false;
          break;
        }
      }
    }
    return retVal;
  }

  HOST_DEVICE
  MatrixCU<T>
  operator*(T const &sca){
    MatrixCU<T> retVal(data.size(), data[0].size());
    for(vsize_t x = 0; x < data.size(); ++x){
      for(vsize_t y = 0; y < data[0].size(); ++y){
        retVal.data[x][y] = data[x][y] * sca;
      }
    }
    return retVal;
  }

  HOST_DEVICE
  MatrixCU<T>
  operator*(MatrixCU<T> const &rhs){
    MatrixCU<T> retVal(data.size(), rhs.data[0].size());
    for(vsize_t bcol = 0; bcol < rhs.data[0].size(); ++bcol){
      for(vsize_t x = 0; x < data.size(); ++x){
        for(vsize_t y = 0; y < data[0].size(); ++y){
          retVal.data[x][bcol] += data[x][y] * rhs.data[y][bcol];
        }
      }
    }
    return retVal;
  }

  HOST_DEVICE
  static
  MatrixCU<T>
  SkewSymCrossProdM(VectorCU<T> const &v){
    MatrixCU<T> retVal(3,3);
    retVal[0][0] = 0;
    retVal[0][1] = -v[2];
    retVal[0][2] = v[1];

    retVal[0][0] = v[2];
    retVal[1][1] = 0;
    retVal[2][2] = -v[0];

    retVal[0][0] = -v[1];
    retVal[1][1] = v[0];
    retVal[2][2] = 0;

    return retVal;
  }

  HOST_DEVICE
  static
  MatrixCU<T>
  Identity(size_t dims){
    MatrixCU<T> retVal(dims, dims);
    for(size_t x = 0; x < dims; ++x){
      for(size_t y = 0; y < dims; ++y){
        if(x == y) retVal[x][y] = 1;
        else retVal[x][y] = 0;
      }
    }
    return retVal;
  }

  HOST_DEVICE
  VectorCU<T>
  operator*(VectorCU<T> const &v) const {
    VectorCU<T> retVal((vsize_t)data.size());
    vsize_t resI = 0;
    for(vsize_t x = 0;
        x < data.size();
        ++x){
      auto row = data[x];
      for(vsize_t i = 0; i < row.size(); ++i){
        retVal[resI] += row[i] * v[i];
      }
      ++resI;
    }
    return retVal;
  }

  HOST_DEVICE
  MatrixCU<T>
  operator+(MatrixCU<T> const&rhs) const{
    MatrixCU<T> retVal(data.size(), data.size());
    int x = 0; int y = 0;
    for(vsize_t j = 0;
        j < data.size();
        ++j){
      auto r = data[j];
      for(vsize_t k = 0;
          k < data.size();
          ++k){
        auto i = r[k];
        retVal[x][y] = i + rhs[x][y];
        ++y;
      }
      y = 0; ++x;
    }
    return retVal;
  }
};

} // end of namespace flit

#endif // MATRIX_CU_H
