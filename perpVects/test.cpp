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

using std::vector;
using std::cout;
using std::endl;
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
class InfoStream : public std::ostream{
  class NullBuffer : public std::streambuf
  {
  public:
    int overflow(int c) { return c; }
  };
public:
  InfoStream():std::ostream(new NullBuffer()){}
  void
  reinit(std::streambuf* b){init(b);}
  void
  show(){reinit(cout.rdbuf());}
  void
  hide(){reinit(new NullBuffer());}
};

//SETTINGS
//NullBuffer null_buffer;
//std::ostream info_stream(&null_buffer);
InfoStream info_stream;
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

  //the first normalized number > 0 (the smallest positive) -- can be obtained from <float>
  //[pos][bias + 1][000...0]
  template<typename T>
  static T
  getFirstNorm(){
    T retVal;
    typedef typename get_corresponding_type<T>::type t;
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
	projectType(val, retVal);
      }
      break;
    case 16:
      {
	val = (t) 1 << mantBitWidth128;
      }
      break;
    default:
      //won't compile
      int x = 0;
    }
    projectType(val, retVal);
    return retVal;
  }

  template<typename T>
  static T
  perturbFP(T const &src, unsigned offset){ //negative offset with 0 may produce NAN
    //    T *ncSrc = const_cast<T*>(&src);
    T ncSrc = src;
    typename get_corresponding_type<T>::type tmp;
    if(NO_SUBNORMALS && ncSrc == 0) ncSrc = getFirstNorm<T>();
    T retVal = perturbFPTyped(ncSrc, tmp, offset);
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
    typename get_corresponding_type<T>::type val;
    projectType(v, val);
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
  getRandomVector(size_t dim, T min_inc, T max_exc,
		  std::default_random_engine::result_type seed = 0){
    gen.seed(seed);
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
    for(int x = 0; x < size(); ++x){
      if(data[x] != b.data[x]){
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

  //method to reduce vector (pre-sort)
  enum sort_t{
    lt, //manual lt magnitude sort
    gt, //manual gt magnitude sort
    bi, //built-in (std::inner_product) [only for ^, assumes def otherwise]
    def //default (manual unsorted)
  };

private:
  sort_t sortType = def;
public:
  template<class U>
  friend ostream& operator<<(ostream& os, Vector<U> const &a);

  //fun is lamda like: [&sum](T a){sum += a};
  //you provide the T sum, closure will capture
  //not const because it may sort cont
  //C is container type
  template<typename F, class C>
  void
  reduce(C &cont, F const &fun) const {
    T retVal;
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
  setSort(sort_t st = def){sortType = st;}
  //inner product (dot product)
  T
  operator^(Vector<T> const &rhs) const {
    T sum = 0;
    if( sortType == bi){
      sum = std::inner_product(data.begin(), data.end(), rhs.data.begin(), 0);
    }else{
      vector<T> prods(data.size());
      for(int i = 0; i < size(); ++i){ 
	prods[i] = data[i] * rhs.data[i]; 
	//	info_stream << "(" << data[i] << " * " << rhs.data[i] << ") = " << prods[i] << endl;
      }
      reduce(prods, [&sum](T e){sum += e;});
    }
    return sum;
  }

  //L2 norm
  T
  L2Norm() const {
    Vector<T> squares(size());
    T retVal = 0;
    vector<T> prods(data); //make a copy of our data for possible sorted reduction
    //will sort reduction if sort
    reduce(prods, [&retVal](T e){retVal += e*e;});
    return sqrt(retVal);
  }

  T
  L2Distance(Vector<T> const &rhs) const {
    T retVal = 0;
    auto diff = operator-(rhs);
    //sorts pre-rediction for sortType = gt|lt
    reduce(diff.data, [&retVal](T e){retVal += e*e;});
    return retVal;
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
	  info_stream << "in: " << __func__ << endl;
	  info_stream << "for x,y: " << x << ":" << y << endl;
	  info_stream << "this = " << data[x][y] << "; rhs = " << rhs.data[x][y] << endl;
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
    info_stream << "entered " << __func__ << endl; 
    long double score = 0.0;
    auto A = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "A (unit) is: " << endl << A << endl;
    auto B = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "B (unit): " << endl  << B << endl;
    auto cross = A * B; //cross product
    info_stream << "cross: " << endl << cross << endl;
    auto sine = cross.L2Norm();
    info_stream << "sine: " << endl << sine << endl;
    auto cosine = A ^ B; //dot product
    info_stream << "cosine: " << endl << cosine << endl;
    auto sscpm = Matrix<T>::SkewSymCrossProdM(cross);
    info_stream << "sscpm: " << endl << sscpm << endl;
    auto rMatrix = Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - cosine)/(sine * sine));
    auto result = rMatrix * A;
    info_stream << "rotator: " << endl << rMatrix << endl;
    if(!(result == B)){
      auto dist = result.L1Distance(B);
      info_stream << "Skew symmetric cross product rotation failed with ";
      info_stream << "L1Distance " << dist << endl;
      info_stream << "starting vectors: " << endl;
      info_stream << A << endl;
      info_stream << "...and..." << endl;
      info_stream << B << endl;
      info_stream << "ended up with: " << endl;
      info_stream << result << endl;
      score = dist;
    }
    return {__func__, score};
  }
  template<typename T, typename Fun>
  static pair<string, long double>
  DoOrthoPerturbTest(const int iters, const int dim,
		     const size_t ulp_inc,
		     Fun f,
		     const typename Vector<T>::sort_t
		     st = Vector<T>::def){
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0);
    size_t indexer = 0;
    Vector<T> a(dim, f);
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
	  if(i != 0) score += p - backup; //score should be perturbed amount
	}else{
	  if(i == 0) score += (a ^ b);  //if falsely not detecting ortho, should be the dot prod
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
    return {__func__, score};
  }

  template<typename T>
  static pair<string, long double>
  DoMatrixMultSanity(size_t dim, T min, T max){
    long double score = 0.0;
    Vector<T> b = Vector<T>::getRandomVector(dim, min, max);
    auto c = Matrix<T>::Identity(dim) * b;
    info_stream << "Product is: " << c << endl;
    bool eq = c == b;
    info_stream << "A * b == b? " << eq << endl;
    if(!eq) score = c.L1Distance(b);
    return {__func__, score};
  }

  template<typename T>
  static pair<string, long double>
  DoSimpleRotate90(){
    long double score = 0.0;
    Vector<T> A = {1, 1, 1};
    Vector<T> expected = {-1, 1, 1};
    info_stream << "Rotating A: " << A << ", 1/4 PI radians " << endl;
    A.rotateAboutZ_3d(M_PI/2);
    info_stream << "Resulting vector: " << A << endl;
    info_stream << "in " << __func__ << endl;
    A.dumpDistanceMetrics(expected, info_stream);
    if(!(A == expected))  score = A.L1Distance(expected);
    return {__func__, score};
  }


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
    auto dist = A.L1Distance(orig);
    if(!equal){
      info_stream << "error in L1 distance is: " << dist << endl;
      info_stream << "difference between: " << (A - orig) << endl;
    }
    info_stream << "in " << __func__ << endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {__func__, dist};
  }

  //this this test creates a random 3d vector, copies it to orig,
  //then rotates n times pi/(n/2) radians, then compares (n * pi/(n/2) = 2pi).
  template <typename T>
  static pair<string, long double>
  RotateFullCircle(size_t n, T min = -1, T max = 1){
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
    if(!equal){
      info_stream << "The (vector) difference is: " << (A - orig) << endl;
      score = A.L1Distance(orig);
    }
    info_stream << "in " << __func__ << endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {__func__, score};
  }
};

namespace UnitTests{
  typedef std::pair<std::string, bool> results;
  //the things we've gotta test:
  //*
  //FPHelpers: getFirstNorm, perturbFP, getExponent
  //*
  //Vector: getUnitVector, genOrthoVector, -, L1Distance,
  //^ (innerproduct), L2Norm, L2Distance, *
  //*
  //Matrix: * (MxM), *(MxV), *(Mxs), +

  template<typename T>
  static results
  TestGenOrthoVector(){
    bool result = true;
    auto A = Vector<T>::getRandomVector(10, -10, 10);
    auto O = A.genOrthoVector();
    if(!(A.isOrtho(O))) result = false;
    return {__func__, result};
  }  

  template<typename T>
  static results
  TestL1Distance(){
    bool result = true;
    Vector<T> A = {12.25, 77.45, 99.9};
    Vector<T> B = {-17.29, 33.3, -1};
    T output = A.L1Distance(B);
    T expected = (12.25 - -17.29) + (77.45 - 33.3) + (99.9 - -1);
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << endl;
      info_stream << "A:" << endl << A << endl;
      info_stream << "B:" << endl << B << endl;
      info_stream << "expected:" << endl << expected << endl;
      info_stream << "output:" << endl << output << endl;
    }
    return{__func__, result};
  }
  
  template<typename T>
  static results
  TestInnerProd(){
    bool result = true;
    Vector<T> A = {12.25, 77.45, 99.9};
    Vector<T> B = {-17.29, 33.3, -1};
    T expected = (12.25*-17.29)+(77.45*33.3)+(99.9*-1);
    T output = A ^ B;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << endl;
      info_stream << "A:" << endl << A << endl;
      info_stream << "B:" << endl << B << endl;
      info_stream << "expected:" << endl << expected << endl;
      info_stream << "output:" << endl << output << endl;
    }
    return{__func__, result};
  }

  
  template<typename T>
  static results
  TestL2Distance(){
    bool result = true;
    Vector<T> A = {2, 3, 4};
    Vector<T> B = {4, 5, 6};
    if(!(A.L2Distance(B) != sqrt(12))) result = false;
    return {__func__, result};
  }
  
  template<typename T>
  static results
  UnitVector(){
    bool result = true;
    auto V = Vector<T>::getRandomVector(5, -20, 20);
    T expected = 1.0;
    auto output = V.getUnitVector().L2Norm();
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << endl;
      info_stream << "V:" << endl << V << endl;
      info_stream << "expected:" << endl << expected << endl;
      info_stream << "output:" << endl << output << endl;
      info_stream << "expected bits: " << std::hex << FPWrap<T>(expected) << endl;
      info_stream << "output bits:" << std::hex << FPWrap<T>(output) << endl;
    }
    return{__func__, result};
  }

  template<typename T>
  static results
  MplusM(){
    bool result = true;
    Matrix<T> A = {{1, 2, 3},
		   {4, 5, 6},
		   {7, 8, 9}};
    Matrix<T> B = {{9, 8, 7},
		   {6, 5, 4},
		   {3, 2, 1}};
    Matrix<T> expected = {{10, 10, 10},
			  {10, 10, 10},
			  {10, 10, 10}};
    if(!(A + B == expected)) result = false;
    return {__func__, result};
  }

  template<typename T>
  static results
  MxV(){
    bool result = true;
    Matrix<T> A = {{77, 16.23, 99},
		   {17.7777, -23.3, 11},
		   {131, 134, 137}};
    Vector<T> b = {-18, 374, 12};
    Vector<T> expected = {5872.02, -8902.199, 49402};
    auto output = A * b;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << endl;
      info_stream << "A:" << endl << A << endl;
      info_stream << "b:" << endl << b << endl;
      info_stream << "expected:" << endl << expected << endl;
      info_stream << "output:" << endl << output << endl;
    }
    return {__func__, result};
  }
  
  template<typename T>
  static results
  MxM(){
    bool result = true;
    Matrix<T> A = {{1,2,3},
		   {.25, .5, .75},
		   {4,5,6}};
    Matrix<T> B = {{7,8,9},
		   {11,12,13},
		   {8,7,5}};
    Matrix<T> expected = {{53, 53, 50},
			  {13.25, 13.25, 12.5},
			  {131, 134, 131}};
    auto output = A * B;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << endl;
      info_stream << "A:" << endl << A << endl;
      info_stream << "B:" << endl << B << endl;
      info_stream << "expected:" << endl << expected << endl;
      info_stream << "output:" << endl << output << endl;
    }
    return {__func__, result};
  
  }
  template<typename T>
  static results
  MxI(){
    bool result = true;
    info_stream << "in " << __func__ << ":" << endl;
    auto ident = Matrix<T>::Identity(3);
    Matrix<T> M = {{1, 2, 3},
		   {4, 5, 6},
		   {7, 8, 9}};
    info_stream << "Multiplying matrices: " << endl;
    info_stream << M << endl;
    info_stream << "and..." << endl;
    info_stream << ident << endl;
    if(M * ident == M) info_stream << "success" << endl;
    else{
      info_stream << "failure with: " << endl;
      info_stream << (M * ident) << endl;
      result = false;
    }
    return {__func__, result};
  }

  //tests Matrix*scal operator
  template<typename T>
  static results
  MxS(){
    bool retVal = true;
    info_stream << "in " << __func__ << ":" << endl;
    Matrix<T> M = {{1,2,3},
		   {4,5,6},
		   {7,8,9}};
    Matrix<T> const target = {{10, 20, 30},
			      {40, 50, 60},
			      {70, 80, 90}};
    info_stream << M << endl;
    T scalar = 10;
    info_stream << "and scalar: " << scalar << endl;
    auto res = M * scalar;
    if(res == target) info_stream << "success" << endl;
    else{
      info_stream << "failed with: " << endl;
      info_stream << res << endl;
      retVal = false;
    }
    return {__func__, retVal};
  }
  bool RunTests(std::map<std::string, bool> &results, bool detailed = false){
    typedef float prec;
    if(detailed) info_stream.show(); //reinit(cout.rdbuf());
    cout << "starting unit tests" << endl;
    results.insert(TestGenOrthoVector<prec>());
    results.insert(TestL1Distance<prec>());
    results.insert(TestL2Distance<prec>());
    results.insert(TestInnerProd<prec>());
    results.insert(UnitVector<prec>());
    results.insert(MplusM<prec>());
    results.insert(MxV<prec>());
    results.insert(MxM<prec>());
    results.insert(MxI<prec>());
    results.insert(MxS<prec>());
  }

  int DoTests(){
    int retVal = 0;
    std::map<std::string, bool> results;
    RunTests(results);
    if(std::any_of(results.begin(), results.end(), [](UnitTests::results x){return x.second;})){
      for(auto i: results){
	cout << i.first << "\t" << i.second << endl;
      }
      retVal = 1;
      cout << "here are the details:" << endl;
      RunTests(results, true);
    }else cout << "all tests passed" << endl;
    return retVal;
  }
}

namespace fpTestSuite{
template<typename T>
void
DoTests(size_t iters,
	size_t highestDim,
	size_t ulp_inc,
	T min,
	T max,
	typename Vector<T>::sort_t reduction_sort_type,
	T theta,
	std::map<string, long double> &scores){
  size_t indexer = 0;
  scores.insert(FPTests::DoOrthoPerturbTest<T>(iters, highestDim,
					       ulp_inc,
					       [&indexer](){return (T)(1 << indexer++);},
					       reduction_sort_type));
					       
  indexer = 0;
  scores.insert(FPTests::DoOrthoPerturbTest<T>(iters, highestDim,
					       ulp_inc, 
		[&indexer](){return 0.2 / pow((T)10.0, indexer++);},
					       reduction_sort_type));
  scores.insert(FPTests::DoMatrixMultSanity<T>(highestDim, min, max));
  scores.insert(FPTests::DoSimpleRotate90<T>());
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
	      int reduction_sort_type,
	      T theta,
	      std::map<string, long double> &scores){
  cout << "*****************************************" << endl;
  cout << "Sub test with:" << endl;
  cout << "precision: " << typeid(T).name() << endl;
  cout << "iters: " << iters << ", max dim: " << highestDim <<
    ", ulp_inc: " << ulp_inc << ", min: " << min << ", max: " << max <<
    ", product sort method: " << getSortName(reduction_sort_type) <<
    ", theta: " << theta << endl;
  for(auto i: scores){
    cout << i.first << ":\t" << i.second << endl;
  }
  long double subtotal = 0;
  for_each(scores.begin(), scores.end(), [&subtotal](std::pair<string, long double> p)
	   {subtotal += p.second;});
  cout << "subtotal score: " << subtotal << endl;
  cout << "*****************************************" << endl;
}

template<typename T>
void
tabulateSubtest(std::map<string, long double> &master,
		std::map<string, long double> &sub){
  for(auto i: sub){
    master[i.first] += i.second;
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
}

int
main(int argc, char* argv[]){
  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  using namespace fpTestSuite;
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
  info_stream.precision(COUT_PREC);

  std::map<string, long double> masterScore;
  std::map<string, long double> scores;
  for(int ipm = 0; ipm < 4; ++ipm){ //reduction sort pre sum
    for(int p = 0; p < 3; ++p){ //float, double, long double
      switch(p){
      case 0: //float
	{
	DoTests<float>(iters, dim, ulp_inc, min, max, getSortT<float>(ipm), theta, scores);
	outputResults<float>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<float>(masterScore, scores);
	break;
	}
      case 1:
	{
	DoTests<double>(iters, dim, ulp_inc, min, max, getSortT<double>(ipm), theta, scores);
	outputResults<double>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<double>(masterScore, scores);
	break;
	}
      case 2:
	{
	DoTests<long double>(iters, dim, ulp_inc, min, max, getSortT<long double>(ipm), theta, scores);
	outputResults<long double>(iters, dim, ulp_inc, min, max, ipm, theta, scores);
	tabulateSubtest<long double>(masterScore, scores);
	break;
	}
      }
    }
  }
  long double mScore = 0;
  std::for_each(masterScore.begin(),
		masterScore.end(),
		[&mScore](pair<string, long double> p){mScore += p.second;});
  cout << "master score is: " << mScore << endl;
  if(mScore != 0) return 1;
}


//this executes unit tests
// int main(){
//   cout.precision(100);
//   using namespace UnitTests;
//   return DoTests();
// }
