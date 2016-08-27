// the header for QFP helpers.  These classes, such as matrix and
// vector, utilize the testBase watch data items for monitoring by
// differential debugging.

#ifndef QFPHELPERS
#define QFPHELPERS

#include "InfoStream.hpp"

#include <ostream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <float.h>
#include <random>
#include <algorithm>
#include <mutex>

#ifndef Q_UNUSED
#define Q_UNUSED(x) (void)x
#endif

#ifdef __CUDA__
#include <cuda.h>
#include <thrust/device_vector>
#include <thrust/host_vector>
#endif

namespace QFPHelpers {

const int RAND_SEED = 1;
extern thread_local InfoStream info_stream;

std::ostream& operator<<(std::ostream&, const unsigned __int128);

namespace FPHelpers {
  inline float
  as_float(uint32_t val) {
    return *reinterpret_cast<float*>(&val);
  }

  inline double
  as_float(uint64_t val) {
     return *reinterpret_cast<double*>(&val);
  }

  inline long double
  as_float(unsigned __int128 val) {
    return *reinterpret_cast<long double*>(&val);
  }

  inline uint32_t
  as_int(float val) {
    return *reinterpret_cast<uint32_t*>(&val);
  }

  inline uint64_t
  as_int(double val) {
    return *reinterpret_cast<uint64_t*>(&val);
  }

  inline unsigned __int128
  as_int(long double val) {
    const unsigned __int128 zero = 0;
    const auto temp = *reinterpret_cast<unsigned __int128*>(&val);
    return temp & (~zero >> 48);
  }
}

template<typename T>
class Matrix;

template <typename T>
class Vector {
  std::vector<T> data;
public:
  Vector():data(0){}
  Vector(std::initializer_list<T> l):data(l){}
  Vector(size_t size):data(size, 0){}
  template <typename U>
  Vector(size_t size, U genFun):data(size, 0){
    std::generate(data.begin(), data.end(), genFun);
  }
  Vector(const Vector &v):data(v.data) {}
  size_t
  size() const {return data.size();}

  T&
  operator[](size_t indx){return data[indx];}

  const T&
  operator[](size_t indx) const {return data[indx];}

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
  getRandomVector(size_t dim, T min_inc, T max_exc,
                  std::mt19937::result_type seed = 0,
                  bool doSeed = true){
    std::mt19937 gen;
    if(doSeed) gen.seed(seed);
    std::uniform_real_distribution<T> dist(min_inc,max_exc);
    Vector<T> tmp(dim);
    for(auto& i: tmp.data){
      i = dist(gen);
    }
    return tmp;
  }

  Vector<T>
  getUnitVector() const {
    Vector<T> retVal(*this);
    return retVal * ((T)1.0 / (this->L2Norm()));
  }

  bool
  operator==(Vector<T> const &b){
    bool retVal = true;
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

  T
  getLInfNorm(){
    T retVal;
    std::pair<int, T> largest;
    for(int x = 0; x < data.size(); ++x){
      T abe = fabs(data[x]);
      if(abe > largest.second){
        largest.first = x;
        largest.second = abe;
      }
    }
  }

  void
  rotateAboutZ_3d(T rads){
    Matrix<T> t = {{(T)cos(rads), (T)-sin(rads), 0},
                   {(T)sin(rads), (T)cos(rads), 0},
                   {0, 0, 1}};
    info_stream << "rotation matrix is: " << t << std::endl;
    Vector<T> tmp(*this);
    tmp = t * tmp;
    info_stream << "in rotateAboutZ, result is: " << tmp << std::endl;
    data = tmp.data;

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
      distance += fabs(data[x] - rhs.data[x]);
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
      T tmp = fabs(e);
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

};
template <typename T>
std::ostream& operator<<(std::ostream& os, Vector<T> const &a){
  for(auto i: a.data){
    os << i << '\t';
  }
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, Matrix<T> const &m){
  for(auto r: m.data){
    for(auto i: r){
      os << i << '\t';
    }
    os << std::endl;
  }
  return os;
}

template<typename T>
class Matrix {
  std::vector<std::vector<T>> data;
public:
  Matrix(unsigned rows, unsigned cols):
    data(rows, std::vector<T>(cols, 0)){}
  Matrix(Matrix<T> const &m):data(m.data){}
  Matrix(std::initializer_list<std::initializer_list<T>> l):
    data(l.size(), std::vector<T>(l.begin()->size())){
    int x = 0; int y = 0;
    for(auto r: l){
      for(auto i: r){
        data[x][y] = i;
        ++y;
      }
      ++x; y = 0;
    }
  }

  friend class Vector<T>;
  template<class U>
  friend std::ostream& operator<<(std::ostream& os, Matrix<U> const &a);


  bool
  operator==(Matrix<T> const &rhs){
    bool retVal = true;
    for(uint x = 0; x < data.size(); ++x){
      for(uint y = 0; y < data[0].size(); ++y){
        if(data[x][y] != rhs.data[x][y]){
          info_stream << "in: " << __func__ << std::endl;
          info_stream << "for x,y: " << x << ":" << y << std::endl;
          info_stream << "this = " << data[x][y] << "; rhs = " << rhs.data[x][y] << std::endl;
          retVal = false;
          break;
        }
      }
    }
    return retVal;
  }

  Matrix<T>
  operator*(T const &sca){
    Matrix<T> retVal(data.size(), data[0].size());
    for(uint x = 0; x < data.size(); ++x){
      for(uint y =0; y < data[0].size(); ++y){
        retVal.data[x][y] = data[x][y] * sca;
      }
    }
    return retVal;
  }

  //precond: this.w = rhs.h, duh
  Matrix<T>
  operator*(Matrix<T> const &rhs){
    Matrix<T> retVal(data.size(), rhs.data[0].size());
    for(uint bcol = 0; bcol < rhs.data[0].size(); ++bcol){
      for(uint x = 0; x < data.size(); ++x){
        for(uint y = 0; y < data[0].size(); ++y){
          retVal.data[x][bcol] += data[x][y] * rhs.data[y][bcol];
        }
      }
    }
    return retVal;
  }

  //precond: dim(v) == 3
  static
  Matrix<T>
  SkewSymCrossProdM(Vector<T> const &v){
    return Matrix<T>(
      {{0, -v[2], v[1]},
       {v[2], 0, -v[0]},
       {-v[1], v[0], 0}});
  }

  static
  Matrix<T>
  Identity(size_t dims){
    Matrix<T> retVal(dims, dims);
    for(size_t x = 0; x < dims; ++x){
      for(size_t y =0; y < dims; ++y){
        if(x == y) retVal.data[x][y] = 1;
        else retVal.data[x][y] = 0;
      }
    }
    return retVal;
  }

  Vector<T>
  operator*(Vector<T> const &v) const {
    Vector<T> retVal(data.size());
    int resI = 0;
    for(auto row: data){
      for(size_t i = 0; i < row.size(); ++i){
        retVal[resI] += row[i] * v[i];
      }
      ++resI;
    }
    return retVal;
  }

  Matrix<T>
  operator+(Matrix<T> const&rhs) const{
    Matrix<T> retVal(rhs);
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

  void
  print() const {
    std::cout << *this;
  }
};

}

#endif
 
