#ifndef CU_VECTOR_HPP
#define CU_VECTOR_HPP

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
//  HOST_DEVICE
//  cuvector&
//  operator=(const cuvector& rhs){
//    if (tsize > 0) delete[] _data;
//    tsize = 0;
//    vsize = rhs.vsize;
//    if (vsize > 0) {
//      _data = new T[vsize];
//      invalid = _data == nullptr;
//      if (!invalid) {
//        for (cvs_t x = 0; x < vsize; ++x) {
//          _data[x] = rhs[x];
//        }
//        tsize=vsize;
//      }
//    }
//    return *this;
//  }
//
//  HOST
//  cuvector&
//  operator=(const std::vector<T>& rhs){
//    // Reuse the move assignment operator and copy constructor
//    *this = cuvector<T>(rhs);
//    return *this;
//    if (tsize > 0) delete[] _data;
//    vsize = rhs.size();
//    if (vsize > 0) {
//      _data = new T[vsize];
//      invalid = _data == nullptr;
//      if(!invalid){
//        for(cvs_t x = 0; x < vsize; ++x){
//          _data[x] = rhs[x];
//        }
//        tsize=vsize;
//      }
//    }
//    return *this;
//  }

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
    printf("hi from emplace\n");
    if(vsize == tsize) grow();
    printf("emp2\n");
    if(!invalid){
      _data[vsize++] = T(std::forward<Args>(args)...);
    }
    printf("emp3\n");
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

#endif // CU_VECTOR_HPP
