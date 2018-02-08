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

#ifndef CUVECTOR_H
#define CUVECTOR_H

#include "CUHelpers.h"

#include <vector>

//This vector class is designed to be used on a CUDA
//enabled device.
//This class needs the following features:
// * uploading data from host
// * array operator
// * begin / end iterators
// * destructor (delete data)
// * constructors:
// * cuvector(std::initializer_list<T>)
// * cuvector(size)
// * cuvector(size, T)
// * cuvector(&cuvector)
// * operator =
// * size()

template <typename T>
class cuvector {
public:
  typedef uint32_t cvs_t;
private:
  T* _data;
  cvs_t vsize; //allocated and assigned
  bool invalid = false; //true when couldn't allocate
  cvs_t tsize; //total allocated
  const cvs_t delta = 10; //grow size

  HOST_DEVICE void zero() { setall(0); }
  HOST_DEVICE
  void setall(T val){
    for(cvs_t i = 0; i < vsize; ++i) {
      _data[i] = val;
    }
  }
public:

  HOST_DEVICE
  cuvector() noexcept : vsize(0),tsize(0) {}

  HOST_DEVICE
  cuvector(cvs_t size):vsize(size),tsize(0){
    _data = new  T[vsize];
    invalid = _data == nullptr;
    if (!invalid) {
      zero();
      tsize = vsize;
    }
  }

  HOST_DEVICE
  cuvector(cvs_t size, T val):vsize(size),tsize(0){
    _data = new T[vsize];
    invalid = _data == nullptr;
    if (!invalid) {
      setall(val);
      tsize = vsize;
    }
  }

  HOST
  cuvector(const std::initializer_list<T> vals):cuvector(){
    for (auto val : vals) {
      push_back(val);
    }
  }

  HOST_DEVICE
  cuvector(const T* array, cvs_t size):vsize(size){
    _data = new T[vsize];
    invalid = _data == nullptr;
    if (!invalid) {
      for(cvs_t x = 0; x < vsize; ++x) {
        _data[x] = array[x];
      }
      tsize = vsize;
    }
  }

  // copy support
  HOST_DEVICE cuvector(const cuvector& rhs) : cuvector(rhs._data, rhs.vsize) {}
  HOST cuvector(const std::vector<T>& rhs) : cuvector(rhs.data(), rhs.size()) {}

  // reuse the move assignment operator and copy constructor
  HOST_DEVICE cuvector& operator=(const cuvector& rhs) { *this = cuvector<T>(rhs); return *this; }
  HOST cuvector& operator=(const std::vector<T>& rhs) { *this = cuvector<T>(rhs); return *this; }

  // move support
  // Unfortunately, we cannot provide moves from std::vector
  // for move constructor, reuse the move assignment operator
  HOST_DEVICE cuvector(cuvector&& rhs) { *this = std::move(rhs); }

  HOST_DEVICE
  cuvector&
  operator=(cuvector&& rhs){
    // Delete the current data
    if (tsize > 0) delete[] _data;
    // Copy it over
    this->vsize = rhs.vsize;
    this->tsize = rhs.tsize;
    this->_data = rhs._data;
    this->invalid = rhs.invalid;
    // Empty the rhs
    rhs.vsize = rhs.tsize = 0;
    rhs.invalid = false;
    rhs._data = nullptr;
    return *this;
  }


  HOST_DEVICE
  ~cuvector(){
    if(tsize > 0) delete[] _data;
  }

  HOST_DEVICE inline T* data() noexcept { return _data; }
  HOST_DEVICE inline const T* data() const noexcept { return _data; }

  HOST_DEVICE
  inline void
  grow(){
    T* temp = new T[tsize + delta];
    if(temp == nullptr)
      invalid = true;
    else{
      for(cvs_t x = 0; x < vsize; ++x){
        temp[x] = _data[x];
      }
      if(tsize > 0) delete[] _data;
      tsize += delta;
      _data = temp;
    }
  }

  HOST_DEVICE
  inline void
  push_back(T val){
    if(vsize == tsize) grow();
    if(!invalid){
      _data[vsize++] = val;
    }
  }

  template<class... Args>
  HOST_DEVICE
  inline void
  emplace_back(Args&&... args){
    if(vsize == tsize) grow();
    if(!invalid){
      _data[vsize++] = T(std::forward<Args>(args)...);
    }
  }

  HOST_DEVICE
  inline
  bool
  isValid() const noexcept {return !invalid;}

  HOST_DEVICE
  inline
  T
  operator[](cvs_t index) const {
    return _data[index];
  }

  HOST_DEVICE
  inline
  T&
  operator[](cvs_t index){
    return _data[index];
  }

  HOST_DEVICE
  inline
  cvs_t
  size() const noexcept {return vsize;}
};

#endif // CUVECTOR_H
