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
 * -- LICENSE END -- */

#ifndef FLIT_VECTOR_H
#define FLIT_VECTOR_H

#include "Matrix.h"
#include "RandHelper.h"

#include <flitHelpers.h>

#include <algorithm>        // for std::generate
#include <cmath>            // for std::sqrt
#include <cstdlib>          // for std::abs
#include <initializer_list> // for std::initializer_list
#include <ostream>          // for std::ostream
#include <random>           // for std::mt19937
#include <utility>          // for std::move
#include <vector>           // for std::vector

template<typename T> class Matrix;

template <typename T>
class Vector {
  std::vector<T> data;
public:
  Vector():data(0){}
  Vector(std::initializer_list<T> l) : data(l) {}
  Vector(size_t size):data(size, 0) {}
  template <typename U>
  Vector(size_t size, U genFun):data(size, 0){
    std::generate(data.begin(), data.end(), genFun);
  }

  // Copyable
  Vector(const Vector &v):data(v.data) {}
  Vector(const std::vector<T> &data) : data(data) {}
  Vector& operator=(const Vector &v) { data = v.data; return *this; }
  Vector& operator=(const std::vector<T> &data) { this->data = data; return *this; }

  // Movable
  Vector(Vector<T> &&v) : data(std::move(v.data)) {}
  Vector(std::vector<T> &&data) : data(std::move(data)) {}
  Vector& operator=(Vector<T> &&v) { data = std::move(v.data); return *this; }
  Vector& operator=(std::vector<T> &&data) { this->data = std::move(data); return *this;}

  size_t size() const { return data.size(); }
  T& operator[](size_t indx){return data[indx];}
  const T& operator[](size_t indx) const {return data[indx];}

  const std::vector<T>& getData() { return data; }

  Vector<T>
  operator*(T const &rhs){
    Vector<T> retVal(size());
    for(uint x = 0; x < size(); ++x){
      retVal[x] = data[x] * rhs;
    }
    return retVal;
  }

  static
  Vector<T>
  getRandomVector(size_t dim){
    auto copy = getRandSeq<T>();
    copy.erase(copy.begin() + dim, copy.end());
    // We need to make a copy of the copy because the first copy is
    // std::vector<float>.  We need std::vector<T>.
    std::vector<T> otherCopy(copy.begin(), copy.end());
    return Vector<T>(std::move(otherCopy));
  }

  Vector<T>
  getUnitVector() const {
    Vector<T> retVal(*this);
    return retVal * ((T)1.0 / (this->L2Norm()));
  }

  bool
  operator==(Vector<T> const &b){
    bool retVal = true;
    if(b.data.size() != this->data.size()) return false;
    for(uint x = 0; x < size(); ++x){
      if(data[x] != b.data[x]){
        retVal = false;
        break;
      }
    }
    return retVal;
  }
  void
  dumpDistanceMetrics(Vector<T> const &b, std::ostream& os){
    os << "vector difference: \t" << ((*this) - b) << std::endl;
    os << "l1distance:  \t" << (*this).L1Distance(b) << std::endl;
  }

  //creates a vector with randomly swapped elements, every other
  //one negated.  For odd sized vectors, the odd one out will
  //be zero
  //i.e.
  // 1  2 3  4
  // may produce
  // 2 -1 4 -3
  //or
  // 1 2  3  4 5
  // may produce
  // 3 4 -1 -2 0

  Vector<T>
  genOrthoVector(){
    Vector<T> retVal(size());
    std::vector<T> seq(size());
    iota(seq.begin(), seq.end(), 0); //load with seq beg w 0

    shuffle(seq.begin(), seq.end(), std::mt19937(RAND_SEED));
    //do pairwise swap
    for(uint i = 0; i < size(); i += 2){
      retVal[seq[i]] = data[seq[i+1]];
      retVal[seq[i+1]] = -data[seq[i]];
    }
    if(size() & 1) //odd
      retVal[seq[size() - 1]] = 0;
    return retVal;
  }

  Vector<T>
  rotateAboutZ_3d(T rads){
    Matrix<T> t = {{(T)cos(rads), (T)-sin(rads), 0},
                   {(T)sin(rads), (T)cos(rads), 0},
                   {0, 0, 1}};
    flit::info_stream << "rotation matrix is: " << t << std::endl;
    Vector<T> tmp(*this);
    tmp = t * tmp;
    flit::info_stream << "in rotateAboutZ, result is: " << tmp << std::endl;
    return tmp;
  }

  Vector<T>
  operator-(Vector<T> const &rhs) const {
    Vector<T> retVal(size());
    for(uint x = 0; x < size(); ++x){
      retVal[x] = data[x] - rhs.data[x];
    }
    return retVal;
  }

  T
  L1Distance(Vector<T> const &rhs) const {
    T distance = 0;
    for(uint x = 0; x < size(); ++x){
      distance += std::abs(data[x] - rhs.data[x]);
    }
    return distance;
  }

  //method to reduce vector (pre-sort)

public:
  template<class U>
  friend std::ostream& operator<<(std::ostream& os, Vector<U> const &a);

  //fun is lamda like: [&sum](T a){sum += a};
  //you provide the T sum, closure will capture
  //not const because it may sort cont
  //C is container type

  template<typename F, class C>
  void
  reduce(C &cont, F const &fun) const {
    for_each(cont.begin(), cont.end(), fun);
  }

  T
  operator^(Vector<T> const &rhs) const {
    T sum = 0.0;
    for(uint i = 0; i < size(); ++i){
      sum += data[i] * rhs.data[i];
    }
    return sum;
  }

  T
  LInfNorm() const {
    T retVal = 0;
    for(auto e: data){
      T tmp = std::abs(e);
      if( tmp > retVal) retVal = tmp;
    }
    return retVal;
  }

  T LInfDistance(Vector<T> const &rhs) const {
    auto diff = operator-(rhs);
    return diff.LInfNorm();
  }

  //L2 norm
  T
  L2Norm() const {
    Vector<T> squares(size());
    T retVal = 0;
    std::vector<T> prods(data);
    reduce(prods, [&retVal](T e){retVal += e*e;});
    return std::sqrt(retVal);
  }

  T
  L2Distance(Vector<T> const &rhs) const {
    T retVal = 0;
    auto diff = operator-(rhs);
    reduce(diff.data, [&retVal](T e){retVal += e*e;});
    return std::sqrt(retVal);
  }

  //cross product, only defined here in 3d
  Vector<T>
  cross(Vector<T> const &rhs) const {
    Vector<T> retVal(size());
    retVal.data[0] = data[1] * rhs.data[2] - rhs.data[1] * data[2];
    retVal.data[1] = rhs.data[0] * data[2] - data[0] * rhs.data[2];
    retVal.data[2] = data[0] * rhs.data[1] - rhs.data[0] * data[1];
    return retVal;
  }

  Vector<T>
  operator*(Vector<T> const &rhs) const {
    Vector<T> retVal(size());
    for(int x = 0; x < size(); ++x){
      retVal[x] = data[x] * rhs.data[x];
    }
    return retVal;
  }

  bool
  isOrtho(Vector<T> const &rhs){
    return operator^(rhs) == (T)0;
  }

}; // end of class Vector

template <typename T>
std::ostream& operator<<(std::ostream& os, Vector<T> const &a){
  for(auto i: a.data){
    os << i << '\t';
  }
  return os;
}

#endif // FLIT_VECTOR_H
