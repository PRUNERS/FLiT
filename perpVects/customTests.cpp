// This is where the tests live for QFP.
// All tests need to derive from TestBase,
// and call registerTest.

#include "testBase.h"
#include "QFPHelpers.h"

#include <cmath>
#include <typeinfo>
#include <iomanip>

#ifndef Q_UNUSED
#define Q_UNUSED(x) (void)x
#endif


//setup for Eigen library test suite
//there's a race on this container -- switching to 1 concurrency
std::map<std::string, QFPTest::resultType> eigenResults;
//QFPTest::resultType eigenResults;
std::mutex eigenResults_mutex;
std::mutex g_test_stack_mutex;
using namespace QFPTest;

//namespace QFPTest {

using namespace QFPHelpers;

template <typename T>
class DoSkewSymCPRotationTest: public TestBase {
public:
  DoSkewSymCPRotationTest(std::string id):TestBase(id){}
  resultType operator()(const testInput& ti) {
    auto& min = ti.min;
    auto& max = ti.max;
    //    auto& crit = getWatchData<T>();
    QFPHelpers::info_stream << "entered " << id << std::endl;
    long double L1Score = 0.0;
    long double LIScore = 0.0;
    auto A = QFPHelpers::Vector<T>::getRandomVector(3, min, max).getUnitVector();
    QFPHelpers::info_stream << "A (unit) is: " << std::endl << A << std::endl;
    auto B = QFPHelpers::Vector<T>::getRandomVector(3, min, max).getUnitVector();
    QFPHelpers::info_stream << "B (unit): " << std::endl  << B << std::endl;
    auto cross = A.cross(B); //cross product
    QFPHelpers::info_stream << "cross: " << std::endl << cross << std::endl;
    auto sine = cross.L2Norm();
    QFPHelpers::info_stream << "sine: " << std::endl << sine << std::endl;
    //    crit = A ^ B; //dot product
    auto cos = A ^ B;
    //    QFPHelpers::info_stream << "cosine: " << std::endl << crit << std::endl;
    QFPHelpers::info_stream << "cosine: " << std::endl << cos << std::endl;
    auto sscpm = QFPHelpers::Matrix<T>::SkewSymCrossProdM(cross);
    QFPHelpers::info_stream << "sscpm: " << std::endl << sscpm << std::endl;
    auto rMatrix = QFPHelpers::Matrix<T>::Identity(3) +
      sscpm + (sscpm * sscpm) * ((1 - cos)/(sine * sine));
    // auto rMatrix = QFPHelpers::Matrix<T>::Identity(3) +
    //   sscpm + (sscpm * sscpm) * ((1 - crit)/(sine * sine));
    auto result = rMatrix * A;
    QFPHelpers::info_stream << "rotator: " << std::endl << rMatrix << std::endl;
    if(!(result == B)){
      L1Score = result.L1Distance(B);
      LIScore = result.LInfDistance(B);
      QFPHelpers::info_stream << "Skew symmetric cross product rotation failed with ";
      QFPHelpers::info_stream << "L1Distance " << L1Score << std::endl;
      QFPHelpers::info_stream << "starting vectors: " << std::endl;
      QFPHelpers::info_stream << A << std::endl;
      QFPHelpers::info_stream << "...and..." << std::endl;
      QFPHelpers::info_stream << B << std::endl;
      QFPHelpers::info_stream << "ended up with: " << std::endl;
      QFPHelpers::info_stream << "L1Distance: " << L1Score << std::endl;
      QFPHelpers::info_stream << "LIDistance: " << LIScore << std::endl;
    }
    return {{{id, typeid(T).name()},
	  {L1Score, LIScore}}};
  }
};

REGISTER_TYPE(DoSkewSymCPRotationTest)

template <typename T>
class DoHariGSBasic: public TestBase {
public:
  DoHariGSBasic(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    Q_UNUSED(ti);

    //auto& crit = getWatchData<T>();
    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
    QFPHelpers::Vector<T> a = {1, e, e};
    QFPHelpers::Vector<T> b = {1, e, 0};
    QFPHelpers::Vector<T> c = {1, 0, e};
    auto r1 = a.getUnitVector();
    //crit = r1[0];
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    //crit =r2[0];
    auto r3 = (c - r1 * (c ^ r1) -
	       r2 * (c ^ r2)).getUnitVector();
    //crit = r3[0];
    T o12 = r1 ^ r2;
    //    crit = o12;
    T o13 = r1 ^ r3;
    //crit = o13;
    T o23 = r2 ^ r3;
    //crit = 023;
    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
      QFPHelpers::info_stream << "in: " << id << std::endl;
      QFPHelpers::info_stream << "applied gram-schmidt to:" << std::endl;
      QFPHelpers::info_stream << "a: " << a << std::endl;
      QFPHelpers::info_stream << "b: " << b << std::endl;
      QFPHelpers::info_stream << "c: " << c << std::endl;
      QFPHelpers::info_stream << "resulting vectors were: " << std::endl;
      QFPHelpers::info_stream << "r1: " << r1 << std::endl;
      QFPHelpers::info_stream << "r2: " << r2 << std::endl;
      QFPHelpers::info_stream << "r3: " << r3 << std::endl;
      QFPHelpers::info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
      QFPHelpers::info_stream << "score (bits): " << QFPHelpers::FPWrap<long double>(score) << std::endl;
      QFPHelpers::info_stream << "score (dec) :" << score << std::endl;
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
  }
};

REGISTER_TYPE(DoHariGSBasic)

template <typename T>
class DoHariGSImproved: public TestBase {
public:
  DoHariGSImproved(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    Q_UNUSED(ti);

    long double score = 0.0;
    T e;
    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
    //matrix = {a, b, c};
    QFPHelpers::Vector<T> a = {1, e, e};
    QFPHelpers::Vector<T> b = {1, e, 0};
    QFPHelpers::Vector<T> c = {1, 0, e};

    auto r1 = a.getUnitVector();
    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
    auto r3 = (c - r1 * (c ^ r1));
    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
    T o12 = r1 ^ r2;
    T o13 = r1 ^ r3;
    T o23 = r2 ^ r3;
    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
      QFPHelpers::info_stream << "in: " << id << std::endl;
      QFPHelpers::info_stream << "applied gram-schmidt to:" << std::endl;
      QFPHelpers::info_stream << "a: " << a << std::endl;
      QFPHelpers::info_stream << "b: " << b << std::endl;
      QFPHelpers::info_stream << "c: " << c << std::endl;
      QFPHelpers::info_stream << "resulting vectors were: " << std::endl;
      QFPHelpers::info_stream << "r1: " << r1 << std::endl;
      QFPHelpers::info_stream << "r2: " << r2 << std::endl;
      QFPHelpers::info_stream << "r3: " << r3 << std::endl;
      QFPHelpers::info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << std::endl;
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
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
    auto& watchPoint = getWatchData<T>();
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0.0);
    //we use a double literal above as a workaround for Intel 15-16
    //compiler bug:
    //https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
    QFPHelpers::Vector<T> a(dim, fun);
    QFPHelpers::Vector<T> b = a.genOrthoVector();

    QFPHelpers::info_stream << "starting dot product orthogonality test with a, b = " << std::endl;
    for(int x = 0; x < dim; ++x) QFPHelpers::info_stream << x << '\t';
    QFPHelpers::info_stream << std::endl;
    QFPHelpers::info_stream << a << std::endl;
    QFPHelpers::info_stream << b << std::endl;
    T backup;

    for(int r = 0; r < dim; ++r){
    T &p = a[r];
      backup = p;
      for(int i = 0; i < iters; ++i){
	//	cout << "r:" << r << ":i:" << i << std::std::endl;
	p = QFPHelpers::FPHelpers::perturbFP(backup, i * ulp_inc);
	//Added this for force watchpoint hits every cycle (well, two).  We shouldn't really
	//be hitting float min
	watchPoint = FLT_MIN;
	watchPoint = a ^ b;

	//DELME debug
	//std::cout << watchPoint << std::endl;

	bool isOrth = watchPoint == 0; //a.isOrtho(b);
	if(isOrth){
	  orthoCount[r]++;
	  if(i != 0) score += fabs(p - backup); //score should be perturbed amount
	}else{
	  if(i == 0) score += fabs(watchPoint); //a ^ b);  //if falsely not detecting ortho, should be the dot prod
	}
	QFPHelpers::info_stream << "i:" << i << ":a[" << r << "] = " << a[r] << ", " << QFPHelpers::FPWrap<T>(a[r]) << " multiplier: " << b[r] << ", " << QFPHelpers::FPWrap<T>(b[r]) << " perp: " << isOrth << " dot prod: " <<
	  QFPHelpers::FPWrap<T>(a ^ b) << std::endl;
      }
      QFPHelpers::info_stream << "next dimension . . . " << std::endl;
      p = backup;
    }
    QFPHelpers::info_stream << "Final report, one iteration set per dimensiion:" << std::endl;
    QFPHelpers::info_stream << '\t' << "ulp increment per loop: " << ulp_inc << std::endl;
    QFPHelpers::info_stream << '\t' << "iterations per dimension: " << iters << std::endl;
    QFPHelpers::info_stream << '\t' << "dimensions: " << dim << std::endl;
    QFPHelpers::info_stream << '\t' << "precision (type): " << typeid(T).name() << std::endl;
    int cdim = 0;
    for(auto d: orthoCount){
      QFPHelpers::info_stream << "For mod dim " << cdim << ", there were " << d << " ortho vectors, product magnitude (biased fp exp): "
			      <<QFPHelpers::FPHelpers::getExponent(a[cdim] * b[cdim]) << std::endl;
      cdim++;
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
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
    QFPHelpers::Vector<T> b = QFPHelpers::Vector<T>::getRandomVector(dim, min, max);
    auto c = QFPHelpers::Matrix<T>::Identity(dim) * b;
    QFPHelpers::info_stream << "Product is: " << c << std::endl;
    bool eq = c == b;
    QFPHelpers::info_stream << "A * b == b? " << eq << std::endl;
    return {{{id, typeid(T).name()}, {c.L1Distance(b), c.LInfDistance(b)}}};
  }
};
REGISTER_TYPE(DoMatrixMultSanity)

template <typename T>
class DoSimpleRotate90: public TestBase {
public:
  DoSimpleRotate90(std::string id):TestBase(id){}

  resultType operator()(const testInput& ti) {
    Q_UNUSED(ti);

    QFPHelpers::Vector<T> A = {1, 1, 1};
    QFPHelpers::Vector<T> expected = {-1, 1, 1};
    QFPHelpers::info_stream << "Rotating A: " << A << ", 1/2 PI radians" << std::endl;
    A.rotateAboutZ_3d(M_PI/2);
    QFPHelpers::info_stream << "Resulting vector: " << A << std::endl;
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(expected, QFPHelpers::info_stream);
    return {{{id, typeid(T).name()}, {A.L1Distance(expected), A.LInfDistance(expected)}}};
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
    auto A = QFPHelpers::Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    QFPHelpers::info_stream << "Rotate and Unrotate by " << theta << " radians, A is: " << A << std::endl;
    A.rotateAboutZ_3d(theta);
    QFPHelpers::info_stream << "Rotated is: " << A << std::endl;
    A.rotateAboutZ_3d(-theta);
    QFPHelpers::info_stream << "Unrotated is: " << A << std::endl;
    bool equal = A == orig;
    QFPHelpers::info_stream << "Are they equal? " << equal << std::endl;
    auto dist = A.L1Distance(orig);
    if(!equal){
      QFPHelpers::info_stream << "error in L1 distance is: " << dist << std::endl;
      QFPHelpers::info_stream << "difference between: " << (A - orig) << std::endl;
    }
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, QFPHelpers::info_stream);
    return {{{id, typeid(T).name()}, {dist, A.LInfDistance(orig)}}};
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
    QFPHelpers::Vector<T> A = QFPHelpers::Vector<T>::getRandomVector(3, min, max);
    auto orig = A;
    T theta = 2 * M_PI / n;
    QFPHelpers::info_stream << "Rotate full circle in " << n << " increments, A is: " << A << std::endl;
    for(int r = 0; r < n; ++r){
      A.rotateAboutZ_3d(theta);
      QFPHelpers::info_stream << r << " rotations, vect = " << A << std::endl;
    }
    QFPHelpers::info_stream << "Rotated is: " << A << std::endl;
    bool equal = A == orig;
    QFPHelpers::info_stream << "Does rotated vect == starting vect? " << equal << std::endl;
    if(!equal){
      QFPHelpers::info_stream << "The (vector) difference is: " << (A - orig) << std::endl;
    }
    QFPHelpers::info_stream << "in " << id << std::endl;
    A.dumpDistanceMetrics(orig, QFPHelpers::info_stream);
    return {{{id, typeid(T).name()}, {A.L1Distance(orig), A.LInfDistance(orig)}}};
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

    // auto& crit = getWatchData<T>();

    // 1/2 b*h = A
    const T checkVal = 0.5 * b * a;  //all perturbations will have the same base and height

    long double score = 0;

    for(T pos = 1; pos <= a; pos += delta){
      auto crit = getArea(a,b,c);
      // crit = getArea(a,b,c);
      b = std::sqrt(std::pow(pos, 2) +
		    std::pow(ti.max, 2));
      c = std::sqrt(std::pow(a - pos, 2) +
		    std::pow(ti.max, 2));
      score += std::abs(crit - checkVal);
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
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
  TrianglePSylv(std::string id):Triangle<T>(id){}

  T
  getArea(const T a,
	  const T b,
	  const T c){
    return (pow(2.0, -2)*sqrt((a+(b+c))*(a+(b-c))*(c+(a-b))*(c-(a-b))));
  }

};

REGISTER_TYPE(TrianglePSylv)

template <typename T>
class DistributivityOfMultiplication : public TestBase {
public:
  DistributivityOfMultiplication(std::string id):TestBase(id){}

  std::vector<std::tuple<T,T,T>> testValues();

  resultType operator()(const testInput& ti) {
    Q_UNUSED(ti);
    auto vals = this->testValues();
    std::vector<T> valuesDistributed;
    std::vector<T> valuesUndistributed;
    resultType returnval;
    for (auto& input : vals) {
      T a = std::get<0>(input);
      T b = std::get<1>(input);
      T c = std::get<2>(input);
      T dist = (a * c) + (b * c);
      T undist = (a + b) * c;
      valuesDistributed.push_back((a * c) + (b * c));
      valuesUndistributed.push_back((a + b) * c);
      QFPHelpers::info_stream << std::setw(8);
      QFPHelpers::info_stream << "DistributivityOfMultiplication: (a,b,c) = (" << a << ","
                              << b << "," << c << ")" << std::endl;
      QFPHelpers::info_stream << "DistributivityOfMultiplication: dist    = "
                              << valuesDistributed.back() << std::endl;
      QFPHelpers::info_stream << "DistributivityOfMultiplication: undist  = "
                              << valuesUndistributed.back() << std::endl;
      returnval.insert({
          {id + "_idx" + std::to_string(returnval.size()), typeid(T).name()},
          {valuesDistributed.back(), valuesUndistributed.back()}
          });
    }
    return returnval;
  }
};

REGISTER_TYPE(DistributivityOfMultiplication)

// Define the inputs
template<>
std::vector<std::tuple<float,float,float>>
DistributivityOfMultiplication<float>::testValues() {
  std::vector<std::tuple<float,float,float>> values;
  auto convert = [](uint32_t x) { return QFPHelpers::FPHelpers::reinterpret_int_as_float(x); };

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  values.emplace_back(convert(0x6b8b4567), convert(0x65ba0c1e), convert(0x49e753d2));
  values.emplace_back(convert(0x233eac52), convert(0x22c1532f), convert(0x2fda27b0));
  values.emplace_back(convert(0x2b702392), convert(0x3280ef92), convert(0x4ece629d));
  values.emplace_back(convert(0x4f78bee7), convert(0x5316ee78), convert(0x4f29be1b));
  values.emplace_back(convert(0x4e27aa59), convert(0x4558b7b6), convert(0x337f4093));
  values.emplace_back(convert(0x0e251a94), convert(0x060ad983), convert(0x702378bd));
  values.emplace_back(convert(0x3321a89c), convert(0x3af748bf), convert(0x602dd168));
  values.emplace_back(convert(0x4e61e16a), convert(0x49f3f8fa), convert(0x03cc52d0));
  values.emplace_back(convert(0x5248c931), convert(0x5da4cce1), convert(0x12384ef7));
  values.emplace_back(convert(0x58a810f3), convert(0x594f3d88), convert(0x649f73f0));
  values.emplace_back(convert(0x07be9118), convert(0x00d2636c), convert(0x6d984f2b));

  return values;
}

template<>
std::vector<std::tuple<double,double,double>>
DistributivityOfMultiplication<double>::testValues() {
  std::vector<std::tuple<double,double,double>> values;
  auto convert = [](uint64_t x) { return QFPHelpers::FPHelpers::reinterpret_int_as_float(x); };

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  values.emplace_back(
      convert(0x7712d691ff8158c1),
      convert(0x7a71b704fdd6a840),
      convert(0x019b84dddaba0d31));
  values.emplace_back(
      convert(0x4443528eec7b4dfb),
      convert(0x43fdcf1f2fb9a656),
      convert(0x7ae870922df32b48));
  values.emplace_back(
      convert(0x5b1a5c8d1177cfaa),
      convert(0x595e670dd52ea7bc),
      convert(0x3df3fa0be1c8a9e4));
  values.emplace_back(
      convert(0x1be04a5a1d07fe8a),
      convert(0x1a506ba2ab6016be),
      convert(0x57980f57f96de4dc));
  values.emplace_back(
      convert(0x4776911a8572ae2e),
      convert(0x47c5c4d1506dcbff),
      convert(0x213ff9f501295930));
  values.emplace_back(
      convert(0x29ac0c261d6b14df),
      convert(0x29fc265909b66aab),
      convert(0x69fbe7786470672b));
  values.emplace_back(
      convert(0x24b22d74fb8d9e6d),
      convert(0x24c1f1083cc4a7f0),
      convert(0x6c494ff916e4714c));
  values.emplace_back(
      convert(0x17d682825d8734bf),
      convert(0x1998785eb236c7ef),
      convert(0x5038e232205d2643));
  values.emplace_back(
      convert(0x3774fe15c207a48d),
      convert(0x3a0371c634d95959),
      convert(0x1cfcc1088ead8d5c));
  values.emplace_back(
      convert(0x622e8170fa214891),
      convert(0x5f1a608b13e2c398),
      convert(0x4e3491b372540b89));

  return values;
}

template<>
std::vector<std::tuple<long double,long double,long double>>
DistributivityOfMultiplication<long double>::testValues() {
  // Here we are assuming that long double represents 80 bits
  std::vector<std::tuple<long double,long double,long double>> values;
  auto convert = [](uint64_t left_half, uint64_t right_half) {
    unsigned __int128 val = left_half;
    val = val << 64;
    val += right_half;
    return QFPHelpers::FPHelpers::reinterpret_int_as_float(val);
  };

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  values.emplace_back(
      convert(0x402b99, 0x2bb4d082ca2e7ec7),  //  3.586714e-1573
      convert(0x40725a, 0x14c0a0cd445b52d5),  //  6.131032e+3879
      convert(0x40075d, 0x0bc91b713fc2fba5)); //  4.278225e-4366
  values.emplace_back(
      convert(0x403408, 0xd98776d83be541b8),  //  1.497721e-922
      convert(0x407da5, 0x32daa5df77e78b5e),  //  2.847787e+4750
      convert(0x40376a, 0xfa52e8946985dab4)); //  8.479921e-662
  values.emplace_back(
      convert(0x402355, 0xb32ca57fbcc68a6c),  //  1.541551e-2209
      convert(0x407337, 0x4855e1d4f174504d),  //  7.201858e+3946
      convert(0x40736a, 0xac4f338d852e88cd)); //  3.863064e+3962
  values.emplace_back(
      convert(0x404727, 0x9c8f934e1cc682d2),  //  3.753403e+551
      convert(0x4002e7, 0x3b753c2d81c6bf78),  //  3.612998e-4709
      convert(0x40485b, 0xeae81af41947d10a)); //  2.936812e+644
  values.emplace_back(
      convert(0x401b91, 0x9e18ddbb66670e9f),  //  4.852600e-2808
      convert(0x40205f, 0x58ab17d3e5309234),  //  5.031688e-2438
      convert(0x404a4a, 0x1b8a0c6541700676)); //  3.521930e+792
  values.emplace_back(
      convert(0x404178, 0xd2cee20bba5c6843),  //  5.069741e+113
      convert(0x403564, 0x58406d82dd970b8e),  //  3.483973e-818
      convert(0x40150a, 0x92cde10402bc42ef)); //  4.292064e-3311
  values.emplace_back(
      convert(0x401965, 0x630847ac1ecd253c),  //  1.288694e-2975
      convert(0x40004c, 0x6e2b4c6a070d3835),  //  1.093228e-4909
      convert(0x40380f, 0x92b14ca6d81b5a24)); //  2.324058e-612
  values.emplace_back(
      convert(0x404492, 0x870e870425dcb0cf),  //  3.384007e+352
      convert(0x4071dd, 0x159330946cecd9a8),  //  1.498527e+3842
      convert(0x40586a, 0xfc38e15fe5d604a5)); //  1.079136e+1882
  values.emplace_back(
      convert(0x40240d, 0xae73609d2bf51b7d),  //  3.680220e-2154
      convert(0x402a67, 0x89b93255d3362c94),  //  8.669256e-1665
      convert(0x402462, 0x79d020dd3c308e90)); //  9.941326e-2129
  values.emplace_back(
      convert(0x406703, 0x6455d50eb2825cf7),  //  3.818039e+3006
      convert(0x401f77, 0x70b75c7169817349),  //  9.267715e-2508
      convert(0x404ab4, 0x27faa40914dad6a6)); //  4.148019e+824

  return values;
}


// namespace adjoint {
// #include "eigen/adjoint.cpp"
// }
// EIGEN_CLASS_DEF(EigenAdjoint, adjoint)
// REGISTER_TYPE(EigenAdjoint)

// namespace array {
// #include "eigen/array.cpp"
// }
// EIGEN_CLASS_DEF(EigenArray, array)
// REGISTER_TYPE(EigenArray)

// namespace array_for_matrix {
// #include "eigen/array_for_matrix.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayForMatrix, array_for_matrix)
// REGISTER_TYPE(EigenArrayForMatrix)

// namespace array_replicate {
// #include "eigen/array_replicate.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayReplicate, array_replicate)
// REGISTER_TYPE(EigenArrayReplicate)

// namespace array_reverse {
//   #include "eigen/array_reverse.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayReverse, array_reverse)
// REGISTER_TYPE(EigenArrayReverse)

// namespace bandmatrix {
//   #include "eigen/bandmatrix.cpp"
// }
// EIGEN_CLASS_DEF(EigenBandMatrix, bandmatrix)
// REGISTER_TYPE(EigenBandMatrix)

// namespace basicstuff {
//   #include "eigen/basicstuff.cpp"
// }
// EIGEN_CLASS_DEF(EigenBasicStuff, basicstuff)
// REGISTER_TYPE(EigenBasicStuff)

// // namespace bicgstab {
// //   #include "eigen/Eigen/src/Core/util/ForwardDeclarations.h"
// //   #include "eigen/bicgstab.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenBicGStab, bicgstab)
// // REGISTER_TYPE(EigenBicGStab)

// namespace block {
//   #include "eigen/block.cpp"
// }
// EIGEN_CLASS_DEF(EigenBlock, block)
// REGISTER_TYPE(EigenBlock)

// namespace cholesky {
//   #include "eigen/cholesky.cpp"
// }
// EIGEN_CLASS_DEF(EigenCholesky, cholesky)
// REGISTER_TYPE(EigenCholesky)

// // namespace cholmod_support {
// //   #include "eigen/cholmod_support.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenCholmodSupport, cholmod_support)
// // REGISTER_TYPE(EigenCholmodSupport)

// namespace commainitializer {
//   #include "eigen/commainitializer.cpp"
// }
// EIGEN_CLASS_DEF(EigenCommaInitializer, commainitializer)
// REGISTER_TYPE(EigenCommaInitializer)

// // namespace conjugate_gradient {
// //   #include "eigen/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h"
// //   #include "eigen/conjugate_gradient.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenConjugateGradient, conjugate_gradient)
// // REGISTER_TYPE(EigenConjugateGradient)

// namespace corners {
//   #include "eigen/corners.cpp"
// }
// EIGEN_CLASS_DEF(EigenCorners, corners)
// REGISTER_TYPE(EigenCorners)

// namespace cwiseop {
//   #include "eigen/cwiseop.cpp"
// }
// EIGEN_CLASS_DEF(EigenCWiseop, cwiseop)
// REGISTER_TYPE(EigenCWiseop)


//we're going to have to isolate eigenMain.h for multiple eigen tests
#include "eigenMain.h"
#include "eigen/unsupported/test/levenberg_marquardt.cpp"
EIGEN_CLASS_DEF(EigenLevenbergMarquardt, levenberg_marquardt)
REGISTER_TYPE(EigenLevenbergMarquardt)

//} //namespace QFPTest
