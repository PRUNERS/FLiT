#include <thrust/device_vector.h>

#include <QFPHelpers.hpp>

namespace CUHelpers{
  using thrust::device_vector;

extern device_vector<float>* cuda_float_rands;
extern device_vector<double>* cuda_double_rands;

template <typename T>
__global__
void loadDeviceData(device_vector<T>* dest, device_vector<T> source);

void
initDeviceData();

// template <typename T>
// __device__
// thrust::device_vector<T>
// getRandSeqCU(){ return thrust::device_vector<T>();}

// template<>
// __device__
// thrust::device_vector<float>
// getRandSeqCU<float>();

// template<>
// __device__
// thrust::device_vector<double>
// getRandSeqCU<double>();

template <typename T>
T
abs(T val){
  if(val > 0) return val;
  else return val * (T)-1;
}

template <typename T>
class MatrixCU;

template <typename T>
class VectorCU {
  thrust::device_vector<T> data;
public:
  __device__  
  VectorCU(size_t dim): data(dim){}

  __device__  
  static
  VectorCU<T>
  getRandomVector(size_t dim){
    VectorCU<T> retVal(dim)
    auto rands = getRandSeqCU<T>();
    for(uint32_t x = 0; x < rands.size(); ++x){
      retVal[x] = rands[x];
    }
    return retVal;
  }

  __device__  
  VectorCU<T>
  getUnitVector() const {
    VectorCU<T> retVal(this->data.size());
    return retVal * ((T)1.0 / (this->L2Norm()));
  }

  __device__  
  bool
  operator==(VectorCU<T> const &b){
    if(this->data.size() != b.data.size()) return false;
    for(uint32_t x = 0; x < b.data.size(); ++x){
      if(this->data[x] != b.data[x]) return false;
    }
    return true;
  }

  __device__  
  T
  L1Distance(VectorCU<T> const &rhs) const {
    T distance = 0;
    for(uint32_t x = 0; x < data.size(); ++x){
      distance += fabs(data[x] - rhs.data[x]);
    }
    return distance;
  }

  __device__  
  T
  operator^(VectorCU<T> const &rhs) const {
    T sum = 0.0;
    for(uint32_t i = 0; i < data.size(); ++x){
      sum += data[i] * rhs.data[i];
    }
    return sum;
  }

  __device__  
  VectorCU<T>
  operator*(VectorCU<T> const &rhs) const{
    VectorCU<T> ret(data.size());
    for(uint32_t x = 0; x < data.size(); ++x){
      ret[x] = data[x] * rhs.data[x];
    }
  }

  __device__  
  T
  LInfNorm() const {
    T largest = 0;
    for(auto x : data){
      T tmp = abs(x);
      if(tmp > largest) largest = tmp;
    }
    return largest;
  }

  __device__  
  T
  LInfDistance(VectorCU<T> const &rhs) const {
    auto diff = operator-(rhs);
    return diff.LInfNorm();
  }

  //this assumes there is only float and double on CUDA (may change for half precision)
  __device__  
  T
  L2Norm() const {
    VectorCU<T> squares = (*this) * (*this);
    T retVal = (T)0.0;
    for(auto i : squares) retVal += i;
    if(sizeof(T) == 4) return sqrtf(retVal);
    else return sqrtd(retVal);
  }

  T
  __device__  
  L2Distance(VectorCU<T> const &rhs) const {
    return L2Norm((*this) - rhs);
  }

  __device__  
  VectorCU<T>
  cross(VectorCU<T> const &rhs) const {
    VectorCU<T> retVal(data.size());
    retVal.data[0] = data[1] * rhs.data[2] - rhs.data[1] * data[2];
    retVal.data[1] = rhs.data[0] * data[2] - data[0] * rhs.data[2];
    retVal.data[2] = data[0] * rhs.data[1] - rhs.data[0] * data[1];
    return retVal;
  }

  __device__  
  bool
  isOrtho(VectorCU<T> const &rhs){
    return operator^(rhs) == (T)0;
  }
};

template<typename T>
class MatrixCU{
  thrust::device_vector<thrust::device_vector<T>> data;
public:
  __device__  
  MatrixCU(uint32_t rows, uint32_t cols):
    data(rows, thrust::device_vector<T>(cols,0)){}

  __device__  
  bool
  operator==(MatrixCU<T> const &rhs) const {
    if(data.size() != rhs.data.size()) return false;
    bool retVal = true;
    for(uint32_t x = 0; x < data.size(); ++x){
      for(uint32_t y = 0; y < data[0].size(); ++y){
        if(data[x][y] != rhs.data[x][y]){
          retVal = false;
          break;
        }
      }
    }
    return retVal;
  }

  __device__  
  MatrixCU<T>
  operator*(T const &sca){
    MatrixCU<T> retVal(data.size(), data[0].size());
    for(uint32_t x = 0; x < data.size(); ++x){
      for(uint32_t y =0; y < data[0].size(); ++y){
        retVal.data[x][y] = data[x][y] * sca;
      }
    }
    return retVal;    
  }
  
  __device__  
  MatrixCU<T>
  operator*(MatrixCU<T> const &rhs){
    MatrixCU<T> retVal(data.size(), rhs.data[0].size());
    for(uint32_t bcol = 0; bcol < rhs.data[0].size(); ++bcol){
      for(uint32_t x = 0; x < data.size(); ++x){
        for(uint32_t y = 0; y < data[0].size(); ++y){
          retVal.data[x][bcol] += data[x][y] * rhs.data[y][bcol];
        }
      }
    }
    return retVal;
  }

  __device__  
  static
  MatrixCU<T>
  SkewSymCrossProdM(VectorCU<T> const &v){
    MatrixCU<T> retVal(3,3);
    retVal.data[0][0] = 0;
    retVal.data[0][1] = -v[2];
    retVal.data[0][2] = v[1];

    retVal.data[0][0] = v[2];
    retVal.data[1][1] = 0;
    retVal.data[2][2] = -v[0];

    retVal.data[0][0] = -v[1];
    retVal.data[1][1] = v[0];
    retVal.data[2][2] = 0;

    return retVal;
  }

  __device__  
  static
  MatrixCU<T>
  Identity(size_t dims){
    MatrixCU<T> retVal(dims, dims);
    for(size_t x = 0; x < dims; ++x){
      for(size_t y = 0; y < dims; ++y){
        if(x == y) retVal.data[x][y] = 1;
        else retVal.data[x][y] = 0;
      }
    }
    return retVal;
  }

  __device__  
  VectorCU<T>
  operator*(VectorCU<T> const &v) const {
    VectorCU<T> retVal(data.size());
    int resI = 0;
    for(auto row: data){
      for(size_t i = 0; i < row.size(); ++i){
        retVal[resI] += row[i] * v[i];
      }
      ++resI;
    }
    return retVal;
  }

  __device__  
  MatrixCU<T>
  operator+(MatrixCU<T> const&rhs) const{
    Matrix<T> retVal(data.size(), data.size());
    int x = 0; int y = 0;
    for(auto r: data){
      for(auto i: r){
        retVal.data[x][y] = i + rhs.data[x][y];
        ++y;
      }
      y = 0; ++x;
    }
    return retVal;
  }
  
};
}

