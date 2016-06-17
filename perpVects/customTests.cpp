// This is where the tests live for QFP.
// All tests need to derive from TestBase,
// and call registerTest.

#include "testBase.h"
#include "QFPHelpers.h"

#include <cmath>
#include <typeinfo>

namespace QFPTest {

using namespace QFPHelpers;

template <typename T>
class DoSkewSymCPRotationTest: public TestBase {
public:
  DoSkewSymCPRotationTest(std::string id):TestBase(id){}
  resultType operator()(const testInput& ti) {
    auto& min = ti.min;
    auto& max = ti.max;
    auto& crit = getWatchData<T>();
    info_stream << "entered " << id << std::endl; 
    long double L1Score = 0.0;
    long double LIScore = 0.0;
    auto A = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "A (unit) is: " << std::endl << A << std::endl;
    auto B = Vector<T>::getRandomVector(3, min, max).getUnitVector();
    info_stream << "B (unit): " << std::endl  << B << std::endl;
    auto cross = A.cross(B); //cross product
    info_stream << "cross: " << std::endl << cross << std::endl;
    auto sine = cross.L2Norm();
    info_stream << "sine: " << std::endl << sine << std::endl;
    crit = A ^ B; //dot product
    info_stream << "cosine: " << std::endl << crit << std::endl;
    auto sscpm = Matrix<T>::SkewSymCrossProdM(cross);
    info_stream << "sscpm: " << std::endl << sscpm << std::endl;
    auto rMatrix = Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - crit)/(sine * sine));
    auto result = rMatrix * A;
    info_stream << "rotator: " << std::endl << rMatrix << std::endl;
    if(!(result == B)){
      L1Score = result.L1Distance(B);
      LIScore = result.LInfDistance(B);
      info_stream << "Skew symmetric cross product rotation failed with ";
      info_stream << "L1Distance " << L1Score << std::endl;
      info_stream << "starting vectors: " << std::endl;
      info_stream << A << std::endl;
      info_stream << "...and..." << std::endl;
      info_stream << B << std::endl;
      info_stream << "ended up with: " << std::endl;
      info_stream << "L1Distance: " << L1Score << std::endl;
      info_stream << "LIDistance: " << LIScore << std::endl;
    }
    return {{id, typeid(T).name()}, {L1Score, LIScore}};
  }
};

REGISTER_TYPE(DoSkewSymCPRotationTest)

template <typename T>
class DoHariGSBasic: public TestBase {
public:
  DoHariGSBasic(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    auto& crit = getWatchData<T>();
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
    Vector<T> a = {1, e, e};
    Vector<T> b = {1, e, 0};
    Vector<T> c = {1, 0, e};
    auto r1 = a.getUnitVector();
    crit = r1[0];
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    crit =r2[0];
    auto r3 = (c - r1 * (c ^ r1) -
	       r2 * (c ^ r2)).getUnitVector();
    crit = r3[0];
    T o12 = r1 ^ r2;
    crit = o12;
    T o13 = r1 ^ r3;
    crit = o13;
    T o23 = r2 ^ r3;
    crit = 023;
    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
      info_stream << "in: " << id << std::endl;
      info_stream << "applied gram-schmidt to:" << std::endl;
      info_stream << "a: " << a << std::endl;
      info_stream << "b: " << b << std::endl;
      info_stream << "c: " << c << std::endl;
      info_stream << "resulting vectors were: " << std::endl;
      info_stream << "r1: " << r1 << std::endl;
      info_stream << "r2: " << r2 << std::endl;
      info_stream << "r3: " << r3 << std::endl;
      info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
      info_stream << "score (bits): " << FPWrap<long double>(score) << std::endl;
      info_stream << "score (dec) :" << score << std::endl;
    }
    return {{id, typeid(T).name()}, {score, 0.0}};
  }
};

REGISTER_TYPE(DoHariGSBasic)

template <typename T>
class DoHariGSImproved: public TestBase {
public:
  DoHariGSImproved(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
    Vector<T> a = {1, e, e};
    Vector<T> b = {1, e, 0};
    Vector<T> c = {1, 0, e};

    auto r1 = a.getUnitVector();
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    auto r3 = (c - r1 * (c ^ r1));
    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
    T o12 = r1 ^ r2;
    T o13 = r1 ^ r3;
    T o23 = r2 ^ r3;
    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
      info_stream << "in: " << id << std::endl;
      info_stream << "applied gram-schmidt to:" << std::endl;
      info_stream << "a: " << a << std::endl;
      info_stream << "b: " << b << std::endl;
      info_stream << "c: " << c << std::endl;
      info_stream << "resulting vectors were: " << std::endl;
      info_stream << "r1: " << r1 << std::endl;
      info_stream << "r2: " << r2 << std::endl;
      info_stream << "r3: " << r3 << std::endl;
      info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
    }
    return {{id, typeid(T).name()}, {score, 0.0}};
  }
};

REGISTER_TYPE(DoHariGSImproved)

template <typename T>
class DoOrthoPerturbTest: public TestBase {
public:
  DoOrthoPerturbTest(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    auto iters = ti.iters;
    auto dim = ti.highestDim;
    size_t indexer = 0;
    auto ulp_inc = ti.ulp_inc;
    auto fun = [&indexer](){return (T)(1 << indexer++);};
    //    auto fun = [&indexer](){return 2.0 / pow((T)10.0, indexer++);};
    auto sort = ti.reduction_sort_type;
    auto& watchPoint = getWatchData<T>();
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0.0);
    //we use a double literal above as a workaround for Intel 15-16
    //compiler bug:
    //https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
    Vector<T> a(dim, fun);
    a.setSort(sort);
    Vector<T> b = a.genOrthoVector();
  
    info_stream << "starting dot product orthogonality test with a, b = " << std::endl;
    for(int x = 0; x < dim; ++x) info_stream << x << '\t';
    info_stream << std::endl;
    info_stream << a << std::endl;
    info_stream << b << std::endl;
    T backup;

    for(int r = 0; r < dim; ++r){
    T &p = a[r];
      backup = p;
      for(int i = 0; i < iters; ++i){
	//	cout << "r:" << r << ":i:" << i << std::std::endl;
	p = FPHelpers::perturbFP(backup, i * ulp_inc);
	//Added this for force watchpoint hits every cycle (well, two).  We shouldn't really
	//be hitting float min
	watchPoint = FLT_MIN;
	watchPoint = a ^ b;
	bool isOrth = watchPoint == 0; //a.isOrtho(b);
	if(isOrth){
	  orthoCount[r]++;
	  if(i != 0) score += fabs(p - backup); //score should be perturbed amount
	}else{
	  if(i == 0) score += fabs(watchPoint); //a ^ b);  //if falsely not detecting ortho, should be the dot prod
	}
	info_stream << "i:" << i << ":a[" << r << "] = " << a[r] << ", " << FPWrap<T>(a[r]) << " multiplier: " << b[r] << ", " << FPWrap<T>(b[r]) << " perp: " << isOrth << " dot prod: " <<
	  FPWrap<T>(a ^ b) << std::endl;
      }
      info_stream << "next dimension . . . " << std::endl;
      p = backup;
    }
    info_stream << "Final report, one iteration set per dimensiion:" << std::endl;
    info_stream << '\t' << "ulp increment per loop: " << ulp_inc << std::endl;
    info_stream << '\t' << "iterations per dimension: " << iters << std::endl;
    info_stream << '\t' << "dimensions: " << dim << std::endl;
    std::string sType = getSortName(sort);
    info_stream << '\t' << "sort: " << sType << std::endl;
    info_stream << '\t' << "precision (type): " << typeid(T).name() << std::endl;
    int cdim = 0;
    for(auto d: orthoCount){
      info_stream << "For mod dim " << cdim << ", there were " << d << " ortho vectors, product magnitude (biased fp exp): "
	   << FPHelpers::getExponent(a[cdim] * b[cdim]) << std::endl;
      cdim++;
    }
    return {{id, typeid(T).name()}, {score, 0.0}};
  }
};

REGISTER_TYPE(DoOrthoPerturbTest)

template <typename T>
class DoMatrixMultSanity: public TestBase {
public:
  DoMatrixMultSanity(std::string id):TestBase(id){}
  
  resultType operator()(const testInput& ti) {
    auto dim = ti.highestDim;
    T min = ti.min;
    T max = ti.max;
    Vector<T> b = Vector<T>::getRandomVector(dim, min, max);
    auto c = Matrix<T>::Identity(dim) * b;
    info_stream << "Product is: " << c << std::endl;
    bool eq = c == b;
    info_stream << "A * b == b? " << eq << std::endl;
    return {{id, typeid(T).name()}, {c.L1Distance(b), c.LInfDistance(b)}};
  }
};
REGISTER_TYPE(DoMatrixMultSanity)

template <typename T>
class DoSimpleRotate90: public TestBase {
public:
  DoSimpleRotate90(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    Vector<T> A = {1, 1, 1};
    Vector<T> expected = {-1, 1, 1};
    info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A.rotateAboutZ_3d(M_PI/2);
    info_stream << "Resulting vector: " << A << std::endl;
    info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, info_stream);
    return {{id, typeid(T).name()}, {A.L1Distance(expected), A.LInfDistance(expected)}};
  }  
};

REGISTER_TYPE(DoSimpleRotate90)

template <typename T>
class RotateAndUnrotate: public TestBase {
public:
  RotateAndUnrotate(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    T min = ti.min;
    T max = ti.max;
    auto theta = M_PI;
    auto A = Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << std::endl;
    A.rotateAboutZ_3d(theta);
    info_stream << "Rotated is: " << A << std::endl;
    A.rotateAboutZ_3d(-theta);
    info_stream << "Unrotated is: " << A << std::endl;
    bool equal = A == orig;
    info_stream << "Are they equal? " << equal << std::endl;
    auto dist = A.L1Distance(orig);
    if(!equal){
      info_stream << "error in L1 distance is: " << dist << std::endl;
      info_stream << "difference between: " << (A - orig) << std::endl;
    }
    info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {{id, typeid(T).name()}, {dist, A.LInfDistance(orig)}};
  }
};

REGISTER_TYPE(RotateAndUnrotate)

template <typename T>
class RotateFullCircle: public TestBase {
public:
  RotateFullCircle(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    auto n = ti.iters;
    T min = ti.min;
    T max = ti.max;
    Vector<T> A = Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    T theta = 2 * M_PI / n;
    info_stream << "Rotate full circle in " << n << " increments, A is: " << A << std::endl;
    for(int r = 0; r < n; ++r){
      A.rotateAboutZ_3d(theta);
      info_stream << r << " rotations, vect = " << A << std::endl;
    }
    info_stream << "Rotated is: " << A << std::endl;
    bool equal = A == orig;
    info_stream << "Does rotated vect == starting vect? " << equal << std::endl;
    if(!equal){
      info_stream << "The (vector) difference is: " << (A - orig) << std::endl;
    }
    info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, info_stream);
    return {{id, typeid(T).name()}, {A.L1Distance(orig), A.LInfDistance(orig)}};
  }
};
  
REGISTER_TYPE(RotateFullCircle)

template <typename T>
class Triangle: public TestBase {
protected:
  
  Triangle(std::string id):TestBase(id){}
  virtual
  T getArea(const T a, const T b, const T c) = 0;
public:
  resultType
  operator()(const testInput& ti){
    T a = ti.max;
    T b = ti.max;
    T c = std::sqrt(std::pow(a,2) + std::pow(b, 2));
    const T delta = ti.max / (T)ti.iters;
    
    auto& crit = getWatchData<T>();

    for(T pos = 0; pos <= a; pos += delta){
      crit += getArea(a,b,c);
      b = std::sqrt(std::pow(pos, 2) +
		    std::pow(ti.max, 2));
      c = std::sqrt(std::pow(a - pos, 2) +
		    std::pow(ti.max, 2));
    }
    long double score = crit;
    return {{id, typeid(T).name()}, {score, 0.0}};
  }
};

template <typename T>
class TrianglePHeron: public Triangle<T> {
public:
  TrianglePHeron(std::string id):Triangle<T>(id){}
  //computes using Heron's formula
  T
  getArea(const T a,
	  const T b,
	  const T c){
    T s = (a + b + c ) / 2;
    return sqrt(s* (s-a) * (s-b) * (s-c));
  }  
};

REGISTER_TYPE(TrianglePHeron)

template <typename T>
class TrianglePSylv: public Triangle<T> {
public:
public:
  TrianglePSylv(std::string id):Triangle<T>(id){}
  
  T
  getArea(const T a,
	  const T b,
	  const T c){
    return (pow(2.0, -2) -
	    2*sqrt((a+(b+c))*(a+(b-c))*(c+(a-b))*(c-(a-b))));
  }

};

REGISTER_TYPE(TrianglePSylv)

  
} //namespace QFPTest
