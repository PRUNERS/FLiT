#ifndef MATRIX_CU_H
#define MATRIX_CU_H

#include "CUHelpers.h"
#include "VectorCU.h"
#include "cuvector.h"

namespace flit {

template<typename T>
class MatrixCU {
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

} // end of namespace flit

#endif // MATRIX_CU_H
