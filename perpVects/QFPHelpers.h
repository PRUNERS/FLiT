// the header for QFP helpers.  These classes, such as matrix and
// vector, utilize the testBase watch data items for monitoring by
// differential debugging.

#ifndef QFPHELPERS
#define QFPHELPERS

#include "InfoStream.h"

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


namespace QFPHelpers {

const int RAND_SEED = 1;
const bool NO_SUBNORMALS = true;
extern thread_local InfoStream info_stream;

void printOnce(std::string, void*);

// returns a bitlength equivalent unsigned type for floats
// and a bitlength equivalent floating type for integral types
template<typename T>
struct get_corresponding_type {
  using type = typename std::conditional_t<
    std::is_floating_point<T>::value && sizeof(T) == 4, uint32_t,
    std::conditional_t<
      std::is_floating_point<T>::value && sizeof(T) == 8, uint64_t,
    std::conditional_t<
      std::is_floating_point<T>::value && sizeof(T) == 16, unsigned __int128,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(float), float,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(double), double,
    std::conditional_t<
      std::is_integral<T>::value && sizeof(T) == sizeof(long double), long double,
    std::conditional_t<
      std::is_same<T, __int128>::value && sizeof(long double) == 16, long double,
    std::conditional_t<
      std::is_same<T, unsigned __int128>::value && sizeof(long double) == 16, long double,
      void
    >>>>>>>>;
};

template <typename T>
using get_corresponding_type_t = typename get_corresponding_type<T>::type;


std::ostream& operator<<(std::ostream&, const unsigned __int128);

namespace FPHelpers {
  const unsigned expBitWidth32 = 8;
  const unsigned expBitWidth64 = 11;
  const unsigned expBitWidth80 = 15;
  const unsigned mantBitWidth32 = FLT_MANT_DIG; //32 - expBitWidth32 - 1;
  const unsigned mantBitWidth64 = DBL_MANT_DIG; //64 - expBitWidth64 - 1;
  const unsigned mantBitWidth80 = LDBL_MANT_DIG; //128 - expBitWidth80 - 1;
  const unsigned bias32 = (1 << (expBitWidth32 - 1)) - 1;
  const unsigned bias64 = (1 << (expBitWidth64 - 1)) - 1;
  const unsigned bias80 = (1 << (expBitWidth80 - 1)) - 1;

  /**
   * Internal method used by reinterpret_convert<> to mask the top bits of an
   * unsigned __int128 after reinterpreting from an 80-bit long double.
   *
   * If the type is anything besides unsigned __int128, then this method does
   * nothing.  Only the __int128 specialization actually does something.
   *
   * If we reinterpret an unsigned __int128 from a long double, which is 80-bit
   * extended, then mask out the higher bits.  This needs to be done because
   * some compilers are leaving garbage in the unused bits (net defined
   * behavior)
   */
  template<typename T> T tryMaskBits81To128(T val) { return val; }

  // This specialization masks all upper bits from 81-128 with zeros
  template<>
  inline unsigned __int128
  tryMaskBits81To128(unsigned __int128 val) {
    const unsigned __int128 zero = 0;
    return val & (~zero >> 48);
  }

  /**
   * Reinterpret float to integral or integral to float
   * 
   * The reinterpreted float value will be an unsigned integral the same size as the passed-in float
   * 
   * The reinterpreted integral value will be a floating point value of the same size
   * 
   * Example:
   *   auto floatVal = reinterpret_convert(0x1f3742ae);
   *   auto intVal   = reinterpret_convert(floatVal);
   *   printf("%e\n", floatVal);
   *   printf("0x%08x\n", intVal);
   * Output:
   *   3.880691e-20
   *   0x1f3742ae
   *
   * @sa reinterpret_int_as_float, reinterpret_float_as_int
   */
  template<typename T>
  get_corresponding_type_t<T> reinterpret_convert(T val) {
    using ToType = get_corresponding_type_t<T>;
    ToType returnVal = *reinterpret_cast<ToType*>(&val);
    // for converting from long double to unsigned __int128, mask out the upper
    // unused bits
    return tryMaskBits81To128(returnVal);
  }

  /** Convenience for reinterpret_convert().  Enforces the type to be integral */
  template<typename I>
  decltype(auto) reinterpret_int_as_float(I val) {
    static_assert(std::is_integral<I>::value
                    || std::is_same<I, __int128>::value
                    || std::is_same<I, unsigned __int128>::value,
                  "Must pass an integral type to reinterpret as floating-point");
    return reinterpret_convert(val);
  }

  /** Convenience for reinterpret_convert().  Enforces the type to be floating-point */
  template<typename F>
  decltype(auto) reinterpret_float_as_int(F val) {
    static_assert(std::is_floating_point<F>::value,
                  "Must pass a floating-point type to reinterpret as integral");
    return reinterpret_convert(val);
  }

  // the first normalized number > 0 (the smallest positive) -- can be obtained from <float>
  // [pos][bias + 1][000...0]
  template<typename T>
  T
  getFirstNorm(){
    static_assert(std::is_floating_point<T>::value,
                  "getFirstNorm() only supports floating point");
    using t = get_corresponding_type_t<T>;
    t val;
    switch(sizeof(T)){
    case 4:
      {
        val = (t)1 << mantBitWidth32;
      }
      break;
    case 8:
      {
        val = (t)1 << mantBitWidth64;
      }
      break;
    case 16:
      {
        val = (t) 1 << mantBitWidth80;
      }
      break;
    }
    return reinterpret_int_as_float(val);
  }

  template<typename T>
  T
  perturbFP(T const &src, uint offset){ //negative offset with 0 may produce NAN
    static_assert(std::is_floating_point<T>::value,
                  "perturbFP() only supports floating point");
    auto retval = reinterpret_float_as_int(src);
    retval += offset;
    return reinterpret_int_as_float(retval);
  }

  //returns the exponent portion of floating point
  template<typename T>
  uint
  getExponent(T v){
    uint retVal = -1;
    auto val = reinterpret_float_as_int(v);
    switch(sizeof(v)){
    case 4:
      {
        retVal = ((val >> (32 - expBitWidth32 - 1) & 0x7F) - bias32);
      }
      break;
    case 8:
      {
        retVal = ((val >> (64 - expBitWidth64 - 1) & 0x7FF) - bias64);
      }
      break;
    case 16:
      {
        retVal = ((val >> (80 - expBitWidth80 - 1) & 0x7FFF) - bias80);
      }
      break;
    default:
      retVal = 0;
    }
    return retVal;
  }
}

template<typename F>
struct FPWrap{
  using I = get_corresponding_type_t<F>;
  F const &floatVal;
  mutable I intVal;
  void
  update() const {
    intVal = FPHelpers::reinterpret_float_as_int(floatVal);
  }

  FPWrap(F const &val):floatVal(val){}
  template<typename U>
  friend std::ostream& operator<<(std::ostream& os, FPWrap<U> const &w);
};

extern std::mutex ostreamMutex;

template <typename U>
std::ostream& operator<<(std::ostream& os, const FPWrap<U> &w){
  w.update();
  ostreamMutex.lock();
  std::ios_base::fmtflags f = os.flags();
  //FIXME can't handle 128 bit values for ostream operations
  os << std::hex << w.intVal;
  os.flags(f);
  ostreamMutex.unlock();
  return os;
}

enum sort_t{
   lt, //manual lt magnitude sort
   gt, //manual gt magnitude sort
   bi, //built-in (std::inner_product) [only for ^, assumes def otherwise]
   def //default (manual unsorted)
};

std::string
getSortName(sort_t val);

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
    shuffle(seq.begin(), seq.end(), std::default_random_engine(RAND_SEED));
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

  unsigned __int128
  BitDistance(Vector<T> const &rhs) const {
    typedef unsigned __int128 itype;
    itype retVal;
    for(uint i = 0; i < size(); ++i){
      retVal += std::labs((itype)FPWrap<long double>((long double)data[i]).intVal -
                          FPWrap<T>((long double)rhs.data[i]).intVal);
    }
    return retVal;
  }

  //may be more useful than L2 as there is no square/sqrt
  //op to lose bits
  T
  L1Distance(Vector<T> const &rhs) const {
    T distance = 0;
    for(uint x = 0; x < size(); ++x){
      distance += fabs(data[x] - rhs.data[x]);
    }
    return distance;
  }

  //method to reduce vector (pre-sort)

private:
  sort_t sortType = def;
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
    if(sortType == lt || sortType == gt){
      if(sortType == lt)
        std::sort(cont.begin(), cont.end(),
                  [](T a, T b){return fabs(a) < fabs(b);});
      else
        std::sort(cont.begin(), cont.end(),
                  [](T a, T b){return fabs(a) > fabs(b);});
    }
    for_each(cont.begin(), cont.end(), fun);
  }

  void
  setSort(sort_t st = def) { sortType = st; }

  T
  operator^(Vector<T> const &rhs) const {
    T sum = 0.0;
    /* auto prods = std::vector<T>(data.size()); */
    /* T temp = 0.0; */
    if( sortType == bi){
      sum = std::inner_product(data.begin(), data.end(), rhs.data.begin(), (T)0.0);
    }else{
      /* std::for_each(prods.begin(), prods.end(), [&](T i){i = 0.0;}); */
      for(uint i = 0; i < size(); ++i){
        sum += data[i] * rhs.data[i];
      }
      /* for(auto j:prods){ */
      /*   temp += j; */
      /* } */
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
    std::vector<T> prods(data); //make a copy of our data for possible sorted reduction
    //will sort reduction if sort
    reduce(prods, [&retVal](T e){retVal += e*e;});
    return std::sqrt(retVal);
  }

  T
  L2Distance(Vector<T> const &rhs) const {
    T retVal = 0;
    auto diff = operator-(rhs);
    //sorts pre-rediction for sortType = gt|lt
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
    info_stream << "in Matrix multiply operator with matrix:" << std::endl;
    info_stream << (*this);
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
