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
