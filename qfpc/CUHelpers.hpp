#pragma once

#if defined(__CPUKERNEL__) || !defined( __CUDA__)
#define HOST_DEVICE
#define HOST
#define DEVICE
#define GLOBAL
#else
#include <cuda.h>
#include <helper_cuda.h>
#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#endif
#include "QFPHelpers.hpp"
#include "CUVector.hpp"

namespace CUHelpers{

template <typename T>
T
csqrt(T val){ return 0;}

template<>
HOST_DEVICE
inline
float
csqrt<float>(float val){return sqrtf(val);}

template<>
HOST_DEVICE
inline
double
csqrt<double>(double val){return sqrt(val);}

template <typename T>
T
cpow(T a, T b){ return 0;}

template<>
HOST_DEVICE
inline
float
cpow<float>(float a, float b){return powf(a,b);}

template<>
HOST_DEVICE
inline
double
cpow<double>(double a, double b){return pow(a,b);}

template <typename T>
T
ccos(T val){ return 0;}

template<>
HOST_DEVICE
inline
float
ccos<float>(float val){return cosf(val);}


template<>
HOST_DEVICE
inline
double
ccos<double>(double val){return cos(val);}

template <typename T>
T
csin(T val){ return 0;}

template<>
HOST_DEVICE
inline
float
csin<float>(float val){return sinf(val);}

template<>
HOST_DEVICE
inline
double
csin<double>(double val){return sin(val);}

void
initDeviceData();

HOST_DEVICE
const float* getRandSeqCU();

HOST_DEVICE
const uint_fast32_t* get16ShuffledCU(); //an array with 0-15 shuffled

template <typename T>
HOST_DEVICE
T
abs(T val){
  if(val > 0) return val;
  else return val * (T)-1;
}

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
  VectorCU(vsize_t dim): data(dim){}

  HOST_DEVICE
  VectorCU&
  operator=(const VectorCU& rhs){
    data = rhs.data;
    return *this;
  }

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
  VectorCU(const VectorCU& rhs):data(rhs.data){}

  HOST_DEVICE  
  static
  VectorCU<T>
  getRandomVector(vsize_t dim){
    VectorCU<T> retVal(dim);
    printf("made retval\n");
    auto rands = getRandSeqCU();
    for(vsize_t x = 0; x < dim; ++x){
      retVal.data[x] = rands[x];
    }
    return retVal;
  }

  //predoncition: this only works with vectors of
  //predetermined size, now 16
  HOST_DEVICE
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
      distance += fabs(data[x] - rhs.data[x]);
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

template<typename T>
class MatrixCU{
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
}

//#endif
