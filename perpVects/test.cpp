// This program perturbs perpendicular vectors to see how many can be detected as not orthogonal
// this supports float, double and long double (128 bit)
// bits from various sources including http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <cmath>
#include <typeinfo>
#include <initializer_list>
#include <type_traits>
#include <map>
#include <float.h>
//#include <cassert>

using std::vector;
using std::cout;
using std::endl;
// using std::hexfloat; //these ostream format modifiers aren't working!  New to gcc 5.0
using std::hex;
using std::dec;
using std::pair;
using std::ostream;
using std::default_random_engine;
using std::iota;
using std::shuffle;
using std::cos;
using std::sin;
using std::fabs;
using std::string;


//used for directing output to nowhere if wanted
class NullBuffer : public std::streambuf
{
public:
  int overflow(int c) { return c; }
};

//SETTINGS
NullBuffer null_buffer;
std::ostream info_stream(&null_buffer);
//#define info_stream cout
typedef float prec; //CHANGE ME, works for float, double, long double
const int D = 16; //must be even
const size_t COUT_PREC = 100;
const size_t FRAC_INC = 1; //ulps to increment per step
const int ITERS = 200; //number of steps
const int RAND_SEED = 1;
const bool STD_DOTP = true;
const bool SORTED_SUM = false; //sorts by magnitude (fabs)
const bool REVERSE_SORT = false;
const bool NO_SUBNORMALS = true;
//END SETTINGS

//returns a bitlength equivalent unsigned type
template<typename T>
struct get_corresponding_type{
  using type = typename std::conditional<
    std::is_floating_point<T>::value && sizeof(T) == 4 , uint32_t,
    typename std::conditional<std::is_floating_point<T>::value && sizeof(T) == 8 , uint64_t,
		     typename std::conditional<std::is_floating_point<T>::value && sizeof(T) == 16 ,
				      unsigned __int128, void>::type>::type>::type;
};

ostream& operator<<(ostream& os, const unsigned __int128 &i){
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
  os << hi << lo;
  return os;
}

struct FPHelpers {
  static const unsigned expBitWidth32 = 8;
  static const unsigned expBitWidth64 = 11;
  static const unsigned expBitWidth128 = 15;
  static const unsigned mantBitWidth32 = FLT_MANT_DIG; //32 - expBitWidth32 - 1;
  static const unsigned mantBitWidth64 = DBL_MANT_DIG; //64 - expBitWidth64 - 1;
  static const unsigned mantBitWidth128 = LDBL_MANT_DIG; //128 - expBitWidth128 - 1;
  static const unsigned bias32 = (1 << expBitWidth32 - 1) - 1;
  static const unsigned bias64 = (1 << expBitWidth64 - 1) - 1;
  static const unsigned bias128 = (1 << expBitWidth128 - 1) - 1;
  
  template<typename S, typename R>
  static void
  projectType(S const &source, R& result){
    S temp = source;
    R* u = reinterpret_cast<R*>(&temp);
    result = *u;
  }
  
  template<typename F>
  static typename get_corresponding_type<F>::type
  projectType(F const &source){
    using I = typename get_corresponding_type<F>::type;
    F temp = source;
    I* u = reinterpret_cast<I*>(&temp);
    return *u;
  }

  //returns the first normalized number > 0
  //[pos][bias + 1][000...0]
  template<typename T>
  static T
  getFirstNorm(){
    T retVal;
    switch(sizeof(T)){
    case 4:
      {
	uint32_t val = (uint32_t)1 << mantBitWidth32;//30;
	//	cout << "here's the created uint: " << hex << val << endl;
	projectType(val, retVal);
      }
      break;
    case 8:
      {
	uint64_t val = (uint64_t)1 << mantBitWidth64;
	projectType(val, retVal);
      }
      break;
    case 16:
      {
	unsigned __int128 val = (unsigned __int128) 1 << mantBitWidth128;
	projectType(val, retVal);
      }
      break;
    default:
      retVal = 0;
    }
    return retVal;
  }

  template<typename T>
  static T
  perturbFP(T const &src, unsigned offset){ //negative offset with 0 gives NAN
    T *ncSrc = const_cast<T*>(&src);
    if(NO_SUBNORMALS && *ncSrc == 0) *ncSrc = getFirstNorm<T>();
    T retVal;
    if(sizeof(*ncSrc) == 8){
      uint64_t tmp;
      retVal = perturbFPTyped(*ncSrc, tmp, offset);
    }else if(sizeof(*ncSrc) == 4){
      uint32_t tmp;
      retVal = perturbFPTyped(*ncSrc, tmp, offset);
    }else{ //128 bit precision
      unsigned __int128 tmp;
      retVal =  perturbFPTyped(*ncSrc, tmp, offset);
    }
    return retVal;
  }

  template<typename T, typename U>
  static T
  perturbFPTyped(T &src, U &tmp, int offset){
    T retVal;
    projectType(src, tmp);
    tmp += offset;
    projectType(tmp, retVal);
    return retVal;
  }

  //returns the exponent portion of floating point
  template<typename T>
  static unsigned
  getExponent(T v){
    unsigned retVal = -1;
    // cout << "sizeof(v) is: " << sizeof(v) << endl;
    switch(sizeof(v)){
    case 4:
      {
	typedef uint32_t t;
	t val;
	projectType(v, val);
	retVal = ((val >> (32 - expBitWidth32 - 1) & 0x7F) - bias32);
      }
      break;
    case 8:
      {
	typedef uint64_t t;
	t val;
	projectType(v, val);
	retVal = ((val >> (64 - expBitWidth64 - 1) & 0x7FF) - bias64); 
      }
      break;
    case 16:
      {
	typedef unsigned __int128 t;
	t val;
	projectType(v, val);
	retVal = ((val >> (128 - expBitWidth128 - 1) & 0x7FFF) - bias128);
      }
      break;
    default:
      retVal = 0;
    }
    return retVal;
  }


};

template<typename F>
struct FPWrap{
  using I = typename get_corresponding_type<F>::type;
  F const &floatVal;
  I intVal;
  void
  update() const {
    FPHelpers::projectType(floatVal, const_cast<FPWrap*>(this)->intVal);
  }

  FPWrap(F const &val):floatVal(val){}
  template<typename U>
  friend ostream& operator<<(ostream& os, FPWrap<U> const &w);
};

template <typename U>
ostream& operator<<(ostream& os, const FPWrap<U> &w){
  w.update();
  std::ios_base::fmtflags f = os.flags();
  //FIXME can't handle 128 bit values for ostream operations
  os << std::hex << w.intVal;
  os.flags(f);
  return os;
}

// template <>
// ostream& operator<<<long double>(ostream& os, const FPWrap<long double> &w){
//   w.update();
//   //  std::ios_base::fmtflags f = std::cout.flags();
//   //FIXME can't handle 128 bit values for ostream operations
//   uint64_t high = w.intVal >> 64;
//   uint64_t low = (uint64_t) w.intVal;
//   os << std::hex << high << low;
//   //  std::cout.flags(f);
//   return os;
// }

//keeping with the traditional mathematics index
//ordering [row][column]
template<typename T>
class Matrix;

std::default_random_engine gen;


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
    for(int x = 0; x < size(); ++x){
      retVal[x] = data[x] * rhs;
    }
    return retVal;
  }

  static
  Vector<T>
  getRandomVector(size_t dim, T min_inc, T max_exc){
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
    return retVal * ((T)1.0 / (this->L2norm()));
  }

  bool
  operator==(Vector<T> const &b){
    bool retVal = true;
    for(int x = 0; x < size(); ++x){
      if(data[x] != b.data[x]){
	//TODO FIXME can't handle 128 bit values on ostream!
	// info_stream << "vector ==, false on i = " << x << " values: " << data[x] <<
	//   ":" << b.data[x] << endl;
	// info_stream << "bits: " << FPWrap<T>(data[x]) << ":" << FPWrap<T>(b.data[x]) << endl;
	// info_stream << "difference: " << data[x] - b.data[x] << endl;
	retVal = false;
	break;
      }
    }
    return retVal;
  }

  void
  dumpDistanceMetrics(Vector<T> const &b, ostream& os){
    os << "vector difference: \t" << ((*this) - b) << endl;
    //    os << "BitDistance: \t" << (*this).BitDistance(b) << endl;
    os << "l1distance:  \t" << (*this).L1Distance(b) << endl;
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
    shuffle(seq.begin(), seq.end(), default_random_engine(RAND_SEED));
    //do pairwise swap
    for(int i = 0; i < size(); i += 2){
      retVal[seq[i]] = data[seq[i+1]];
      retVal[seq[i+1]] = -data[seq[i]];
    }
    if(size() & 1) //odd
      retVal[seq[size() - 1]] = 0;
    return retVal;
  }

  //returns the angle between vectors using cos^-1 and dot
  // T
  // getTransAngle(Vector<T> const &other){
  //   //return acos(
  // }

  void
  rotateAboutZ_3d(T rads){
    Matrix<T> t = {{cos(rads), -sin(rads), 0},
		{sin(rads), cos(rads), 0},
		{0, 0, 1}};
    info_stream << "rotation matrix is: " << t << endl;
    Vector<T> tmp(*this);
    tmp = t * tmp;
    info_stream << "in rotateAboutZ, result is: " << tmp << endl;
    data = tmp.data;
    
  }

  Vector<T>
  operator-(Vector<T> const &rhs) const {
    Vector<T> retVal(size());
    for(int x = 0; x < size(); ++x){
      retVal[x] = data[x] - rhs.data[x];
    }
    return retVal;
  }

  unsigned __int128
  BitDistance(Vector<T> const &rhs) const {
    typedef unsigned __int128 itype;
    itype retVal;
    for(int i = 0; i < size(); ++i){
      retVal += std::labs((itype)FPWrap<long double>((long double)data[i]).intVal - FPWrap<T>((long double)rhs.data[i]).intVal);
    }
    return retVal;
  }

  //may be more useful than L2 as there is no square/sqrt
  //op to lose bits
  T
  L1Distance(Vector<T> const &rhs) const {
    T distance = 0;
    for(int x = 0; x < size(); ++x){
      distance += fabs(data[x] - rhs.data[x]);
    }
    return distance;
  }

  //dot product specifiers
  enum sort_t{
    lt, //manual lt magnitude sort
    gt, //manual gt magnitude sort
    bi, //built-in (std::inner_product)
    def //default (manual unsorted)
  };

private:
  sort_t sortType = def;
public:
  template<class U>
  friend ostream& operator<<(ostream& os, Vector<U> const &a);
  
  void
  setSort(sort_t st = def){sortType = st;}
  //inner product (dot product)
  T
  operator^(Vector<T> const &rhs){
    //    assert(const_cast<size_t>(size()) == rhs.size() && "Mismatched vector sizes in ^");
    T sum = 0;
    if( sortType == bi){
      sum = std::inner_product(data.begin(), data.end(), rhs.data.begin(), 0);
    }else{
      vector<T> prods(data.size());
      for(int i = 0; i < size(); ++i){ 
	prods[i] = data[i] * rhs.data[i]; //is this promoted to FMA? (need support for FMA)
	info_stream << "(" << data[i] << " * " << rhs.data[i] << ") = " << prods[i] << endl;
      }
      if( sortType == gt ){
	std::sort(prods.begin(), prods.end(), [](T l, T r){ return fabs(l) > fabs(r);});
      }else if(sortType == lt){
	std::sort(prods.begin(), prods.end(), [](T l, T r){ return fabs(l) < fabs(r);});
      }
      info_stream << "list of products to sum:"<< endl;
      for(auto p: prods) info_stream << p << " ";
      info_stream << endl;
      std::for_each(prods.begin(), prods.end(), [&sum](T p){sum += p;});
      info_stream << "dot prod is: " << sum << endl;
    }
    return sum;
  }

  //L2 norm
  T
  L2norm() const {
    Vector<T> squares(size());
    T retVal = 0;
    for(int x = 0; x < size(); ++x){
      retVal += data[x] * data[x];
    }
    return sqrt(retVal);
  }
 
  //cross product
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
ostream& operator<<(ostream& os, Vector<T> const &a){
  for(auto i: a.data){
    os << i << '\t';
  }
  return os;
}

template <typename T>
ostream& operator<<(ostream& os, Matrix<T> const &m){
  for(auto r: m.data){
    for(auto i: r){
      os << i << '\t';
    }
    os << endl;
  }
  return os;
}
  
template<typename T>
class Matrix {
  vector<vector<T>> data;
public:
  Matrix(unsigned rows, unsigned cols):
    data(rows, vector<T>(cols, 0)){}
  Matrix(Matrix<T> const &m):data(m.data){}
  Matrix(std::initializer_list<std::initializer_list<T>> l):
    data(l.size(), vector<T>(l.begin()->size())){
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
  friend ostream& operator<<(ostream& os, Matrix<U> const &a);


  bool
  operator==(Matrix<T> const &rhs){
    bool retVal = true;
    for(int x = 0; x < data.size(); ++x){
      for(int y = 0; y < data[0].size(); ++y){
	if(data[x][y] != rhs.data[x][y]){
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
    for(int x = 0; x < data.size(); ++x){
      for(int y =0; y < data[0].size(); ++y){
	retVal.data[x][y] = data[x][y] * sca;
      }
    }
    return retVal;
  }
  
  //precond: this.w = rhs.h, duh
  Matrix<T>
  operator*(Matrix<T> const &rhs){
    Matrix<T> retVal(data.size(), rhs.data[0].size());
    for(int bcol = 0; bcol < rhs.data[0].size(); ++bcol){
      for(int x = 0; x < data.size(); ++x){
	for(int y = 0; y < data[0].size(); ++y){
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
    for(int x = 0; x < dims; ++x){
      for(int y =0; y < dims; ++y){
	if(x == y) retVal.data[x][y] = 1;
	else retVal.data[x][y] = 0;
      }
    }
    return retVal;
  }
  
  Vector<T>
  operator*(Vector<T> const &v) const {
    info_stream << "in Matrix multiply operator with matrix:" << endl;
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
    cout << *this;
  }
};


struct FPTests {

  //each test returns a score, the closer to 0 the better

  //this test takes two random vectors, calculates the required
  //rotation matrix for alignment, and then attempts to align them
  //the score is the distance between the two after the process
  template<typename T>
  static pair<string, long double>
  DoSkewSymCPRotationTest(const T min, const T max){
    cout << "entered " << __PRETTY_FUNCTION__ << endl; 
    long double score = 0.0;
    auto A = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    cout << "A (unit) is: " << endl << A << endl;
    auto B = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    cout << "B (unit): " << endl  << B << endl;
    auto cross = A * B; //cross product
    cout << "cross: " << endl << cross << endl;
    auto sine = cross.L2norm();
    cout << "sine: " << endl << sine << endl;
    auto cosine = A ^ B; //dot product
    cout << "cosine: " << endl << cosine << endl;
    auto sscpm = Matrix<T>::SkewSymCrossProdM(cross);
    cout << "sscpm: " << endl << sscpm << endl;
    auto rMatrix = Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - cosine)/(sine * sine));
    auto result = rMatrix * A;
    cout << "rotator: " << endl << rMatrix << endl;
    if(!(result == B)){
      auto dist = result.L1Distance(B);
      cout << "Skew symmetric cross product rotation failed with ";
      cout << "L1Distance " << dist << endl;
      cout << "starting vectors: " << endl;
      cout << A << endl;
      cout << "...and..." << endl;
      cout << B << endl;
      cout << "ended up with: " << endl;
      cout << result << endl;
      score = dist;
    }
    return {__PRETTY_FUNCTION__, score};
  }
  template<typename T>
  static pair<string, long double>
  DoOrthoPerturbTest(const int iters, const int dim,
		     const size_t ulp_inc,
		     const typename Vector<T>::sort_t
		     st = Vector<T>::def){
    //    typedef typename get_corresponding_type<T>::type itype;
    //unsigned __int128 score = 0;
    long double score = 0.0;
    // itype score = 0;
    std::vector<unsigned> orthoCount(dim, 0);
    Vector<T> a(dim, [](){ static unsigned x = 0; return (T)(1 << x++);});
    a.setSort(st);
    Vector<T> b = a.genOrthoVector();
  
    info_stream << "starting dot product orthogonality test with a, b = " << endl;
    for(int x = 0; x < dim; ++x) info_stream << x << '\t';
    info_stream << endl;
    info_stream << a << endl;
    info_stream << b << endl;
    T backup;
    for(int r = 0; r < dim; ++r){
    T &p = a[r];
      backup = p;
      for(int i = 0; i < iters; ++i){
	p = FPHelpers::perturbFP(backup, i * ulp_inc);
	bool isOrth = a.isOrtho(b);
	if(isOrth){
	  orthoCount[r]++;
	  if(i != 0) score += a.L1Distance(b);
	}else{
	  if(i == 0) score += a.L1Distance(b);
	}
	info_stream << "a[" << r << "] = " << a[r] << " perp: " << isOrth << endl;
      }
      info_stream << "next dimension . . . " << endl;
      p = backup;
    }
    info_stream << "Final report, one iteration set per dimensiion:" << endl;
    info_stream << '\t' << "ulp increment per loop: " << ulp_inc << endl;
    info_stream << '\t' << "iterations per dimension: " << iters << endl;
    info_stream << '\t' << "dimensions: " << dim << endl;
    string sType;
    switch(st){
    case Vector<T>::def:
      sType = "none";
      break;
    case Vector<T>::gt:
      sType = "decreasing";
      break;
    case Vector<T>::bi:
      sType = "std::inner_product";
      break;
    case Vector<T>::lt:
      sType = "increasing";
    }
    info_stream << '\t' << "sort: " << sType << endl;
    info_stream << '\t' << "precision (type): " << typeid(T).name() << endl;
    int cdim = 0;
    for(auto d: orthoCount){
      info_stream << "For mod dim " << cdim << ", there were " << d << " ortho vectors, product magnitude (biased fp exp): "
	   << FPHelpers::getExponent(a[cdim] * b[cdim]) << endl;
      cdim++;
    }
    return {__PRETTY_FUNCTION__, score};
  }

  //this takes two orthogonal 3d vectors, a theta 'start' and
  //an divisor.  I'ts hoped that the passed in
  //theta will be sufficient to detect non-orthogonality,
  //and then 'start' will be divided until orthogonality is
  //detected again
  // template<typename T>
  // static void
  // getFirstOrthoTheta(Vector<T> const &a, Vector<T> const &b,
  // 		     T &start, T div){
  //   Vector<T> tmp = a;
  //   tmp.rotateAboutZ_3d(start);
  //   if(tmp.isOrtho(b)){
  //     cout << "initial theta, " << start << ", passed to " <<
  // 	__PRETTY_FUNCTION__ << " does not rotate 'a' to a detectable non-ortho position from b" << endl;
  //     exit(1);
  //   }
  //   do{
  //     start /= div;
  //     tmp = a;
  //     tmp.rotateAboutZ_3d(start);
  //   } while(tmp.isOrtho(b));
  //   info_stream << "First ortho theta is: " << start << endl;
  // }
  
  // template<typename T>
  // static void
  // DoVectRotate3dTest(const int iters, const size_t ulp_inc,
  // 		     const int st = Vector<T>::def){
  //   size_t orthoCount = 0;
  //   const int dim = 3;
  //   Vector<T> a = {3, 4, 1};
  //   a.setSort(Vector<T>::bi);
  //   Vector<T> b = {3, -2, -1}; //ortholgonal 3d vectors
  //   cout << "starting dot product orthogonality test after rotation with a, b = " << endl;
  //   for(int x = 0; x < dim; ++x) cout << x << '\t';
  //   cout << endl;
  //   cout << a << endl;
  //   cout << b << endl;
  //   T rrads = (T)M_PI/4;
  //   // using corUint = get_corresponding_type<T>::type;
  //   FPWrap<T> rWrap(rrads);
  //   Vector<T> rotated = a;
  //   getFirstOrthoTheta(a, b, rrads, (T)1.05);
  //   info_stream << "rrads is now:\t(bits) " << rWrap << endl;
  //   info_stream << "\t(value): " << rrads << endl;
  //   for(int i = 0; i < iters; ++i){
  //     rrads = FPHelpers::perturbFP(rrads, i * ulp_inc);
  //     info_stream << "rrads is now: " << rWrap << endl;
  //     rotated.rotateAboutZ_3d(rrads);
  //     bool isOrth = a.isOrtho(b);
  //     if(isOrth) ++orthoCount;
  //     info_stream << "a = " << a << " isOrtho = " << isOrth << endl;
  //     rotated = a;
  //   }
  //   cout << "Final report, ortho detection on infinitismal rotation of one of pair" << endl;
  //   cout << '\t' << "ulp increment per loop: " << ulp_inc << endl;
  //   cout << '\t' << "iterations: " << std::dec << iters << endl;
  //   cout << '\t' << "dimensions: " << dim << endl;
  //   cout << '\t' << "total ortho: " << orthoCount << endl;
    
  // }

  //for this test, we take ortho 3d vectors,
  //test ortho, compute rotation matrix to
  //align, rotate, and then calculate angle (s/b 0)
  // static void
  // AlignAndCompare(){
  // }
  template<typename T>
  static pair<string, long double>
  DoMatrixMultSanity(size_t dim, T min, T max){
    // typedef typename get_corresponding_type<T>::type itype;
    // typedef unsigned __int128 itype;
    // itype score = 0;
    long double score = 0.0;
    Vector<T> b = Vector<T>::getRandomVector(dim, min, max);
    auto c = Matrix<T>::Identity(dim) * b;
    info_stream << "Product is: " << c << endl;
    bool eq = c == b;
    info_stream << "A * b == b? " << eq << endl;
    if(!eq) score = c.L1Distance(b);
    return {__PRETTY_FUNCTION__, score};
  }

  template<typename T>
  static pair<string, long double>
  DoSimpleRotate90(){
    // typedef typename get_corresponding_type<T>::type itype;
    // typedef unsigned __int128 itype;
    // itype score = 0;
    long double score = 0.0;
    Vector<T> A = {1, 1, 1};
    Vector<T> expected = {-1, 1, 1};
    info_stream << "Rotating A: " << A << ", 1/4 PI radians " << endl;
    A.rotateAboutZ_3d(M_PI/2);
    info_stream << "Resulting vector: " << A << endl;
    //DELME
    cout << "in " << __PRETTY_FUNCTION__ << endl;
    A.dumpDistanceMetrics(expected, info_stream);
    if(!(A == expected))  score = A.L1Distance(expected);
    return {__PRETTY_FUNCTION__, score};
  }

  // template<typename T>
  // static pair<string, T>
  // Rotate90CheckOrtho(){
  //   Vector<T> A = {1, 1, 1};
  //   auto B = A;
  //   cout << "Rotating A: " << A << ", 1/2 PI radians " << endl;
  //   A.rotateAboutZ_3d(M_PI);
  //   cout << "Resulting vector: " << A << endl;
  //   cout << "ortho? " << A.isOrtho(B) << endl;
  //   return {__PRETTY_FUNCTION__, s
  // }

  // template<typename T>
  // static pair<string, T>
  // Rotate3dZUntilOrtho(size_t iters){
  //   Vector<T> A = {1, 1, 1};
  //   auto B = Vector<T>({1, 1, -1});
  //   auto n = M_PI / 2 / iters;
  //   info_stream << "Rotating A: " << A <<
  //     ", PI / " << iters << " radians until ortho to original + 0, 0, -2" << endl;
  //   size_t count = 0;
  //   do{
  //     A.rotateAboutZ_3d((T)M_PI / iters);
  //     ++count;
  //   } while (!(A.isOrtho(B)) && count < iters);
  //   info_stream << "Resulting vector: " << A << endl;
  //   info_stream << "ortho? " << A.isOrtho(B) << endl;
  //   //DELME
  //   cout << "in " << __PRETTY_FUNCTION__ << endl;
  //   A.dumpDistanceMetrics(expected);

  //   return {__PRETTY_FUNCTION__, A.BitDistance(B)};
  // }

  //this test takes a random 3d vector, rotates it and then
  //rotates it again with a theta negative of the previous (unrotates) and
  //then checks their distance (
  template <typename T>
  static pair<string, long double>
  RotateAndUnrotate(T min = -1.0, T max = 1.0, T theta = M_PI){
    auto A = Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << endl;
    A.rotateAboutZ_3d(theta);
    info_stream << "Rotated is: " << A << endl;
    A.rotateAboutZ_3d(-theta);
    info_stream << "Unrotated is: " << A << endl;
    bool equal = A == orig;
    info_stream << "Are they equal? " << equal << endl;
    // auto dist = A.BitDistance(orig);
    auto dist = A.L1Distance(orig);
    if(!equal){
      info_stream << "error in L1 distance is: " << dist << endl;
      info_stream << "difference between: " << (A - orig) << endl;
    }
    //DELME
    cout << "in " << __PRETTY_FUNCTION__ << endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {__PRETTY_FUNCTION__, dist};
  }

  //this this test creates a random 3d vector, copies it to orig,
  //then rotates n times pi/(n/2) radians, then compares (n * pi/(n/2) = 2pi).
  template <typename T>
  static pair<string, long double>
  RotateFullCircle(size_t n, T min = -1, T max = 1){
    // typedef typename get_corresponding_type<T>::type itype;
    // typedef unsigned __int128 itype;
    // itype score = 0;
    long double score = 0.0;
    Vector<T> A = Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    prec theta = 2 * M_PI / n;
    info_stream << "Rotate full circle in " << n << " increments, A is: " << A << endl;
    for(int r = 0; r < n; ++r){
      A.rotateAboutZ_3d(theta);
      info_stream << r << " rotations, vect = " << A << endl;
    }
    info_stream << "Rotated is: " << A << endl;
    bool equal = A == orig;
    info_stream << "Does rotated vect == starting vect? " << equal << endl;
    // itype bitDistance = 0;
    if(!equal){
      // auto bitDistance = A.BitDistance(orig);
      // std::ios_base::fmtflags f = std::cout.flags();
      // info_stream << "The bit distance is: " << std::hex << bitDistance << endl;
      // std::cout.flags(f);
      info_stream << "The (vector) difference is: " << (A - orig) << endl;
      //      score = bitDistance;
      score = A.L1Distance(orig);
    }
    //DELME
    cout << "in " << __PRETTY_FUNCTION__ << endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {__PRETTY_FUNCTION__, score};
  }

  // static void
  // SimpleRotate(){
  //   Vector<float> a = {1, 1, 1};
  //   float theta = M_PI;
  //   cout << "starting test rotating a 3d vector " << theta << " radians about z" << endl;
  //   for(int x = 0; x < 100; ++x){
  //     a.rotateAboutZ_3d(theta);
  //     cout << "x:" << x << "\t" << a << endl;
  //   }
  // }

  static void
  SimpleMatrixMultTest(){
    cout << "doing simple test of matrix * matrix and matrix * scalar" << endl;
    auto ident = Matrix<float>::Identity(3);
    Matrix<float> M = {{1, 2, 3},
		       {4, 5, 6},
		       {7, 8, 9}};
    cout << "Multiplying matrices: " << endl;
    cout << M << endl;
    cout << "and..." << endl;
    cout << ident << endl;
    if(M * ident == M) cout << "success" << endl;
    else{
      cout << "failure with: " << endl;
      cout << (M * ident) << endl;
    }
  }

  static void
  SimpleMatrixScalarMultTest(){
    cout << "simple scalar mult test with matrix: " << endl;
    Matrix<float> M = {{1,2,3},
		       {4,5,6},
		       {7,8,9}};
    Matrix<float> const target = {{10, 20, 30},
				  {40, 50, 60},
				  {70, 80, 90}};
    cout << M << endl;
    float scalar = 10;
    cout << "and scalar: " << scalar << endl;
    auto res = M * scalar;
    if(res == target) cout << "success" << endl;
    else{
      cout << "failed with: " << endl;
      cout << res << endl;
    }
  }
};

template<typename T>
void
DoTests(size_t iters,
	size_t highestDim,
	size_t ulp_inc,
	T min,
	T max,
	typename Vector<T>::sort_t inner_prod_method,
	T theta,
	std::map<string, long double> &scores){
	// std::map<string, unsigned __int128> &scores){
  scores.insert(FPTests::DoOrthoPerturbTest<T>(iters, highestDim,
				      ulp_inc, inner_prod_method));
  scores.insert(FPTests::DoMatrixMultSanity<T>(highestDim, min, max));
  scores.insert(FPTests::DoSimpleRotate90<T>());
  //  scores.insert(FPTests::Rotate3dZUntilOrtho<T>(iters));
  scores.insert(FPTests::RotateAndUnrotate<T>(min, max, theta));
  scores.insert(FPTests::RotateFullCircle<T>(iters, min, max));
  scores.insert(FPTests::DoSkewSymCPRotationTest(min, max));
}


string
getSortName(int val){
  switch(val){
  case 0:
    return "less than";
  case 1:
    return "greater than";
  case 2:
    return "built-in (inner_product)";
  case 3:
    return "default (unsorted)";
  default:
    return "something bad happened, undefined sort type";
  }
}
  
template<typename T>
void
outputResults(size_t iters,
	      size_t highestDim,
	      size_t ulp_inc,
	      T min,
	      T max,
	      int inner_prod_method,
	      T theta,
	      std::map<string, long double> &scores){
	      // std::map<string, unsigned __int128> &scores){
  cout << "*****************************************" << endl;
  cout << "Sub test with:" << endl;
  cout << "precision: " << typeid(T).name() << endl;
  cout << "iters: " << iters << ", max dim: " << highestDim <<
    ", ulp_inc: " << ulp_inc << ", min: " << min << ", max: " << max <<
    ", product sort method: " << getSortName(inner_prod_method) <<
    ", theta: " << theta << endl;
  for(auto i: scores){
    cout << i.first << ":\t" << i.second << endl;
  }
  long double subtotal = 0;
  // unsigned __int128 subtotal = 0;
  for_each(scores.begin(), scores.end(), [&subtotal](std::pair<string, long double> p)
  // for_each(scores.begin(), scores.end(), [&subtotal](std::pair<string, unsigned __int128> p)
	   {subtotal += p.second;});
  cout << "subtotal score: " << subtotal << endl;
  cout << "*****************************************" << endl;
}

template<typename T>
void
tabulateSubtest(std::map<string, long double> &master,
		std::map<string, long double> &sub){
// tabulateSubtest(std::map<string, unsigned __int128> &master,
// 		std::map<string, unsigned __int128> &sub){
  for(auto i: sub){
    master[i.first] += i.second;
    //    FPHelpers::projectType(FPWrap<T>(master[i.first]).intVal + FPWrap<T>(i.second).intVal, master[i.first]);
  }
  sub.clear();
}

template<typename T>
typename Vector<T>::sort_t
getSortT(int t){
  switch(t){
  case 0:
    return Vector<T>::lt;
  case 1:
    return Vector<T>::gt;
  case 2:
    return Vector<T>::bi;
  case 3:
    return Vector<T>::def;
  default:
    return Vector<T>::def;
  }
}

int
main(int argc, char* argv[]){
  // The params to perturb are:
  // precision := {float | double | long double}
  // innerIterations := 200
  size_t iters = 200;
  // highestDim := 16
  size_t dim = 16;
  // ulp_inc := 1
  size_t ulp_inc = 1;
  // randElementMin
  float min = -6.0;
  float max = 6.0;
  // randElementMax
  // dot_product_method := Vector<prec>::{def, lt, gt, bi}
  // theta [for rotation tests]
  float theta = M_PI;
  
  cout.precision(COUT_PREC); //set cout to print many decimal places
  int dotProductMethod = Vector<prec>::def; //can be def, lt, gt, bi
  const int TestCount = 6;

  std::map<string, long double> masterScore;
  std::map<string, long double> scores;
  // std::map<string, unsigned __int128> masterScore;
  // std::map<string, unsigned __int128> scores;
  for(int ipm = 0; ipm < 4; ++ipm){ //inner product sort pre sum
    for(int p = 0; p < 3; ++p){ //float, double, long double
      switch(p){
      case 0: //float
	{
	// std::map<string, float>scores;
	DoTests<float>(iters, dim, ulp_inc, min, max, getSortT<float>(ipm), theta, scores);
	outputResults<float>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<float>(masterScore, scores);
	break;
	}
      case 1:
	{
	// std::map<string, double>scores;
	DoTests<double>(iters, dim, ulp_inc, min, max, getSortT<double>(ipm), theta, scores);
	outputResults<double>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<double>(masterScore, scores);
	break;
	}
      case 2:
	{
	// std::map<string, long double>scores;
	DoTests<long double>(iters, dim, ulp_inc, min, max, getSortT<long double>(ipm), theta, scores);
	outputResults<long double>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<long double>(masterScore, scores);
	break;
	}
      }
    }
  }
  long double mScore = 0;
  // unsigned __int128 mScore = 0;
  std::for_each(masterScore.begin(),
		masterScore.end(),
		[&mScore](pair<string, long double> p){mScore += p.second;});
		// [&mScore](pair<string, unsigned __int128> p){mScore += p.second;});
  cout << "master score is: " << mScore << endl;
  // unsigned __int128 retVal;
  // FPHelpers::projectType(mScore, retVal);
  // return retVal;
  return mScore;
}

// int main(){
//   FPTests::SimpleMatrixMultTest();
//   FPTests::SimpleMatrixScalarMultTest();
// }
