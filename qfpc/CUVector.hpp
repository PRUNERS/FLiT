#pragma once
#include "CUHelpers.hpp"

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

template <typename T>
class cuvector {
public:
  typedef uint32_t cvs_t;
private:
  T* data;
  cvs_t vsize; //allocated and assigned
  bool invalid = false; //true when couldn't allocate
  cvs_t tsize; //total allocated
  const cvs_t delta = 10; //grow size
  
  HOST_DEVICE
  void
  zero(){
    for(cvs_t x = 0; x < vsize; ++x)
      data[x] = 0;
  }
public:

  HOST_DEVICE
  cuvector():vsize(0),tsize(0){}

  HOST_DEVICE
  cuvector(cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    if (!invalid) zero();
    tsize=vsize;
  }
  
  HOST_DEVICE
  cuvector(cvs_t size, T val):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
        data[x] = val;
      }
      tsize=vsize;
    }
  }

  HOST_DEVICE
  cuvector(const cuvector& rhs):vsize(rhs.vsize){
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
	data[x] = rhs.data[x];
      }
      tsize=vsize;
    }
  }
  
  HOST_DEVICE
  cuvector&
  operator=(const cuvector& rhs){
    vsize = rhs.vsize;
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
	data[x] = rhs.data[x];
      }
      tsize=vsize;
    }
    return *this;
  }
  
  HOST_DEVICE
  cuvector(T* array, cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
	data[x] = array[x];
      }
      tsize=vsize;
    }
  }

  HOST_DEVICE
  ~cuvector(){
    delete[] data;
  }

  HOST_DEVICE
  inline void
  grow(){
    T* temp = new T[tsize + delta];
    if(temp == NULL)
      invalid = true;
    else{
      for(cvs_t x = 0; x < vsize; ++x){
	temp[x] = data[x];
      }
      if(tsize > 0) delete[] data;
      tsize += delta;
      data = temp;
    }
  }

  HOST_DEVICE
  inline void
  push_back(T val){
    if(vsize == tsize) grow();
    if(!invalid){
      data[vsize++] = val;
    }
  }
    
  template<class... Args>
  HOST_DEVICE
  inline void
  emplace_back(Args&&... args){
    printf("hi from emplace\n");
    if(vsize == tsize) grow();
    printf("emp2\n");
    if(!invalid){
      data[vsize++] = T(std::forward<Args>(args)...);
    }
    printf("emp3\n");
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
