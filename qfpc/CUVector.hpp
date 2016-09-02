#pragma once
#include "CUHelpers.hpp"

//#include <cuda.h>

//This vector class is designed to be used on a CUDA
//enabled device.
//This class needs the following features:
// * uploading data from host
// * array operator
// * begin / end iterators
// * destructor (delete data)
// * constructors:
// * cuvector(size)
// * cuvector(size, T)
// * cuvector(&cuvector)
// * operator =
// * size()

//#define HOST_DEVICE
//#define HOST_DEVICE __host__ __device__

template <typename T>
class cuvector {
public:
  typedef uint32_t cvs_t;
private:
  T* data;
  cvs_t vsize;
  bool invalid = false;

  HOST_DEVICE
  void
  zero(){
    for(cvs_t x = 0; x < vsize; ++x)
      data[x] = 0;
  }
public:

  HOST_DEVICE
  cuvector():vsize(0){}

  HOST_DEVICE
  cuvector(cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    if (!invalid) zero();
  }
  
  HOST_DEVICE
  cuvector(cvs_t size, T val):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
        data[x] = val;
      }
    }
  }

  HOST_DEVICE
  cuvector(const cuvector& rhs):vsize(rhs.vsize){
    data = new  T[vsize];
    invalid = data == NULL;
    for(cvs_t x = 0; x < vsize; ++x){
      data[x] = rhs.data[x];
    }
  }
  
  HOST_DEVICE
  cuvector&
  operator=(const cuvector& rhs){
    vsize = rhs.vsize;
    data = new  T[vsize];
    invalid = data == NULL;
    for(cvs_t x = 0; x < vsize; ++x){
      data[x] = rhs.data[x];
    }
    return *this;
  }
  
  HOST_DEVICE
  cuvector(T* array, cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    for(cvs_t x = 0; x < vsize; ++x){
      data[x] = array[x];
    }
  }

  HOST_DEVICE
  ~cuvector(){
    delete[] data;
  }

  HOST_DEVICE
  inline
  bool
  isValid() const {return !invalid;}

  HOST_DEVICE
  inline
  T
  operator[](cvs_t index) const {
    return data[index];
  }

  HOST_DEVICE
  inline
  T&
  operator[](cvs_t index){
    return data[index];
  }

  HOST_DEVICE
  inline
  cvs_t
  size() const {return vsize;}
};
