#include <cuda.h>

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
  cvs_t vsize;
  bool invalid = false;
public:

  __host__ __device__
  cuvector():vsize(0){}

  __host__ __device__
  cuvector(cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
  }
  
  __host__ __device__
  cuvector(cvs_t size, T val){
    data = new  T[vsize];
    invalid = data == NULL;
    if(!invalid){
      for(cvs_t x = 0; x < vsize; ++x){
        data[x] = val;
      }
    }
  }

  __host__ __device__
  cuvector(const cuvector& rhs):vsize(rhs.vsize){
    data = new  T[vsize];
    invalid = data == NULL;
    for(cvs_t x = 0; x < vsize; ++x){
      data[x] = rhs.data[x];
    }
  }
  
  __host__ __device__
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
  
  __host__ __device__
  cuvector(T* array, cvs_t size):vsize(size){
    data = new  T[vsize];
    invalid = data == NULL;
    for(cvs_t x = 0; x < vsize; ++x){
      data[x] = array[x];
    }
  }

  __host__ __device__
  ~cuvector(){
    delete[] data;
  }

  __host__ __device__
  inline
  bool
  isValid() const {return !invalid;}

  __host__ __device__
  inline
  T
  operator[](cvs_t index) const {
    return data[index];
  }

  __host__ __device__
  inline
  T&
  operator[](cvs_t index){
    return data[index];
  }

  __host__ __device__
  inline
  cvs_t
  size() const {return vsize;}
};
