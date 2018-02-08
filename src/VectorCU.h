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

#ifndef VECTOR_CU_H
#define VECTOR_CU_H

#include "CUHelpers.h"
#include "cuvector.h"

namespace flit {

template <typename T>
class MatrixCU;

template <typename T>
class VectorCU {
  cuvector<T> data;
  friend class MatrixCU<T>;
public:
  using vsize_t = typename cuvector<T>::cvs_t;

  HOST_DEVICE
  explicit
  VectorCU(vsize_t dim) : data(dim) {}
  HOST VectorCU(std::initializer_list<T> l) : data(l) {}
  HOST_DEVICE VectorCU(const T* array, vsize_t size) : data(array, size) {}

  // copy support
  HOST_DEVICE VectorCU(const VectorCU& rhs):data(rhs.data){}
  HOST_DEVICE VectorCU(const cuvector<T>& vals):data(vals){}
  HOST_DEVICE VectorCU& operator=(const VectorCU& rhs) { data = rhs.data; return *this; }
  HOST_DEVICE VectorCU& operator=(const cuvector<T>& vals) { data = vals; return *this; }

  // move support
  HOST_DEVICE VectorCU(VectorCU&& rhs):data(std::move(rhs.data)){}
  HOST_DEVICE VectorCU(cuvector<T>&& vals):data(std::move(vals)){}
  HOST_DEVICE VectorCU& operator=(VectorCU&& rhs) { data = std::move(rhs.data); return *this; }
  HOST_DEVICE VectorCU& operator=(cuvector<T>&& vals) { data = std::move(vals); return *this; }

  HOST_DEVICE
  T&
  operator[](vsize_t index){
    return data[index];
  }

  HOST_DEVICE
  T
  operator[](vsize_t index) const {
    return data[index];
  }

  HOST_DEVICE
  inline vsize_t
  size() const noexcept {
    return data.size();
  }

  DEVICE
  static
  VectorCU<T>
  getRandomVector(vsize_t dim){
    VectorCU<T> retVal(dim);
    auto rands = getRandSeqCU();
    for(vsize_t x = 0; x < dim; ++x){
      retVal.data[x] = rands[x];
    }
    return retVal;
  }

  //predoncition: this only works with vectors of
  //predetermined size, now 16
  DEVICE
  VectorCU<T>
  genOrthoVector() const {
    VectorCU<T> retVal(data.size());
    auto shuff = get16ShuffledCU();
    for(vsize_t x = 0; x < data.size(); x += 2){
      retVal[shuff[x]] = data[shuff[x+1]];
      retVal[shuff[x+1]] = -data[shuff[x]];
    }
    return retVal;
  }

  HOST_DEVICE
  VectorCU<T>
  rotateAboutZ_3d(T rads){
    MatrixCU<T> t(3,3);
    t[0][0]=ccos(rads); t[0][1]=-csin(rads); t[0][2]=0;
    t[1][0]=csin(rads); t[1][1]=ccos(rads);  t[1][2]=0;
    t[2][0]=0;          t[2][1]=0;           t[2][2]=1;
    return t * (*this);
  }

  HOST_DEVICE
  VectorCU<T>
  getUnitVector() const {
    VectorCU<T> retVal(*this);
    return retVal * ((T)1.0 / (L2Norm()));
  }

  HOST_DEVICE
  bool
  operator==(VectorCU<T> const &b){
    if(this->data.size() != b.data.size()) return false;
    for(vsize_t x = 0; x < b.data.size(); ++x){
      if(data[x] != b.data[x]) return false;
    }
    return true;
  }

  HOST_DEVICE
  T
  L1Distance(VectorCU<T> const &rhs) const {
    T distance = 0;
    for(vsize_t x = 0; x < data.size(); ++x){
      distance += std::abs(data[x] - rhs.data[x]);
    }
    return distance;
  }

  HOST_DEVICE
  T
  operator^(VectorCU<T> const &rhs) const {
    T sum = 0.0;
    for(vsize_t i = 0; i < data.size(); ++i){
      sum += data[i] * rhs.data[i];
    }
    return sum;
  }

  HOST_DEVICE
  VectorCU<T>
  operator*(VectorCU<T> const &rhs) const{
    VectorCU<T> ret(data.size());
    for(vsize_t x = 0; x < data.size(); ++x){
      ret[x] = data[x] * rhs.data[x];
    }
    return ret;
  }

  HOST_DEVICE
  VectorCU<T>
  operator*(T const& sca) const {
    VectorCU<T> ret(data.size());
    for(vsize_t x = 0; x < data.size(); ++x){
      ret[x] = data[x] * sca;
    }
    return ret;
  }

  HOST_DEVICE
  VectorCU<T>
  operator-(const VectorCU<T>& rhs) const {
    VectorCU<T> retVal(data.size());
    for(vsize_t x = 0;
        x < data.size();
        ++x){
      retVal.data[x] = data[x] - rhs.data[x];
    }
    return retVal;
  }

  HOST_DEVICE
  T
  LInfNorm() const {
    T largest = 0;
    for(vsize_t x = 0;
        x < data.size();
        ++x){
      T tmp = abs(data[x]);
      if(tmp > largest) largest = tmp;
    }
    return largest;
  }

  HOST_DEVICE
  T
  LInfDistance(VectorCU<T> const &rhs) const {
    auto diff = operator-(rhs);
    return diff.LInfNorm();
  }

  //TODO this assumes there is only float and double on
  //CUDA (may change for half precision)
  HOST_DEVICE
  T
  L2Norm() const {
    VectorCU<T> squares = (*this) * (*this);
    T retVal = (T)0.0;
    for(vsize_t x = 0;
        x < data.size();
        ++x) retVal += squares.data[x];
    if(sizeof(T) == 4) return sqrtf(retVal);
    else return sqrt(retVal);
  }

  T
  HOST_DEVICE
  L2Distance(VectorCU<T> const &rhs) const {
    return ((*this) - rhs).L2Norm();
  }

  HOST_DEVICE
  VectorCU<T>
  cross(VectorCU<T> const &rhs) const {
    VectorCU<T> retVal(data.size());
    retVal.data[0] = data[1] * rhs.data[2] - rhs.data[1] * data[2];
    retVal.data[1] = rhs.data[0] * data[2] - data[0] * rhs.data[2];
    retVal.data[2] = data[0] * rhs.data[1] - rhs.data[0] * data[1];
    return retVal;
  }

  HOST_DEVICE
  bool
  isOrtho(VectorCU<T> const &rhs){
    return operator^(rhs) == (T)0;
  }
};

} // end of namespace flit

#endif // VECTOR_CU_H
