#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"
#include "testBase.hpp"

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <type_traits>

using namespace QFPHelpers;

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class FtoDecToF: public QFPTest::TestBase<T> {
public:
  FtoDecToF(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
  QFPTest::TestInput<T> ti;
    ti.vals = {std::nextafter((T)0.0, (T)1.0)};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    std::numeric_limits<T> nlim;
    // from https://en.wikipedia.org/wiki/IEEE_floating_point
    uint16_t ddigs = nlim.digits * std::log10(2) + 1;
    std::ostringstream res;
    res << std::setprecision(ddigs) << ti.vals[0];
    std::string dstr;
    dstr = res.str();
    T backAgain;
    std::istringstream(dstr) >> backAgain;
    return{std::fabs((long double)ti.vals[0] - backAgain), 0.0};
  }

  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(FtoDecToF)

// template <typename T>
// GLOBAL
// void
// subnormalKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class subnormal: public QFPTest::TestBase<T> {
public:
  subnormal(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {std::nextafter((T)0.0, (T)1.0)};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    return {ti.vals[0] - ti.vals[0] / 2, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(subnormal)

// template <typename T>
// GLOBAL
// void
// dotProdKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class dotProd: public QFPTest::TestBase<T> {
public:
  dotProd(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {0.0}; //dummy val -- we load the vectors in
    //run_impl
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Q_UNUSED(ti);
    auto size = 16;

    auto rand = getRandSeq<T>();

    Vector<T> A(std::vector<T>(rand.begin(),
			    rand.begin() + size));
    Vector<T> B(std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size));
    return {A ^ B, 0.0};
    
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(dotProd)

// template <typename T>
// GLOBAL
// void
// simpleReductionKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class simpleReduction: public QFPTest::TestBase<T> {
public:
  simpleReduction(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {0.0}; //dummy
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Q_UNUSED(ti);
    auto vals = getRandSeq<T>();
    auto sublen = vals.size() / 4 - 1;
    T sum = 0;
    for(uint32_t i = 0; i < sublen; i += 4){
      sum += vals[i];
      sum += vals[i+1];
      sum += vals[i+2];
      sum += vals[i+3];
    }
    for(uint32_t i = sublen; i < vals.size(); ++i){
      sum += vals[i];
    }
    return {(long double) sum, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(simpleReduction)

//This test adds L1 + L2 + s, where L1 & L2 are large, s small

// template <typename T>
// GLOBAL
// void
// addTOLKernel(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class addTOL : public QFPTest::TestBase<T> {
public:
  addTOL(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(man_bits + 1, nls.max_exponent);
    //generate the range of offsets, and then generate the
    //mantissa bits for each of the three inputs
    auto L1e = dis(gen); //L1 exponent

    //for the ldexp function we're using, it takes an unbiased exponent and
    //there is no implied 1 MSB for the mantissa / significand
    T zero = (T)0.0;
    auto L1m = as_int(zero);
    auto L2m = as_int(zero);
    auto sm = as_int(zero);
    for(int i = 0; i < man_bits; ++i){
      L1m &= (gen() & 1) << i;
      L2m &= (gen() & 1) << i;
      sm  &= (gen() & 1) << i;
    }
    ti.vals = {
      std::ldexp(as_float(L1m), L1e),
      std::ldexp(as_float(L2m), L1e - 1),
      std::ldexp(as_float(sm), L1e - man_bits)
    };
    return ti;
  }
protected:
  virtual
    QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
    QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] + ti.vals[1] + ti.vals[2];
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};

//the basic idea of this test is A(I) + B + TOL, where A & B are large,
// and TOL is tiny.  
REGISTER_TYPE(addTOL)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class addSub: public QFPTest::TestBase<T> {
public:
  addSub(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)1.0};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    auto big = std::pow(2, (T)man_bits - 1);
    auto res = (ti.vals[0] + big) - big;
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(addSub)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class divc: public QFPTest::TestBase<T> {
public:
  divc(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] / ti.vals[1];
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(divc)
// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class zeroMinusX: public QFPTest::TestBase<T> {
public:
  zeroMinusX(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = (T)0.0 - ti.vals[0];
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(zeroMinusX)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xMinusZero: public QFPTest::TestBase<T> {
public:
  xMinusZero(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] - (T)0.0;
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xMinusZero)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class zeroDivX: public QFPTest::TestBase<T> {
public:
  zeroDivX(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = (T)0.0 / ti.vals[0];
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(zeroDivX)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xDivOne: public QFPTest::TestBase<T> {
public:
  xDivOne(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] / (T)1.0;
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xDivOne)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xDivNegOne: public QFPTest::TestBase<T> {
public:
  xDivNegOne(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] / (T)-1.0;
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xDivNegOne)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class negAdivB: public QFPTest::TestBase<T> {
public:
  negAdivB(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = -(ti.vals[0] / ti.vals[1]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(negAdivB)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

// template <typename T>
// class twiceCast: public QFPTest::TestBase<T> {
// public:
//   twiceCast(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

//   virtual size_t getInputsPerRun() { return 1; }
//   virtual QFPTest::TestInput<T> getDefaultInput(){
//     QFPTest::TestInput<T> ti;
//     ti.vals = {(T)getRandSeq<T>()[0]};
//     return ti;
//   }
// protected:
//   virtual
//   QFPTest::KernelFunction<T>* getKernel() {return NULL; }
//   virtual
//   QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
//     //yes, this is ugly.  ti.vals s/b vector of floats
//     auto res = (T)((std::result_of<::get_next_type(T)>::type)ti.vals[0]);
//     return {res, 0.0};
//   }
//   using QFPTest::TestBase<T>::id;
// };
// REGISTER_TYPE(twiceCast)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class negAminB: public QFPTest::TestBase<T> {
public:
  negAminB(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = -(ti.vals[0] - ti.vals[1]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(negAminB)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xMinusX: public QFPTest::TestBase<T> {
public:
  xMinusX(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {(T)getRandSeq<T>()[0]};
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] - ti.vals[0];
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xMinusX)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class negAplusB: public QFPTest::TestBase<T> {
public:
  negAplusB(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = -(ti.vals[0] + ti.vals[1]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(negAplusB)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class aXbDivC: public QFPTest::TestBase<T> {
public:
  aXbDivC(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1],
      (T)getRandSeq<T>()[2]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] * (ti.vals[1] / ti.vals[2]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(aXbDivC)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class aXbXc: public QFPTest::TestBase<T> {
public:
  aXbXc(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1],
      (T)getRandSeq<T>()[2]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] * (ti.vals[1] * ti.vals[2]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(aXbXc)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class aPbPc: public QFPTest::TestBase<T> {
public:
  aPbPc(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)getRandSeq<T>()[1],
      (T)getRandSeq<T>()[2]
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] + (ti.vals[1] + ti.vals[2]);
    return {res, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(aPbPc)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xPc1EqC2: public QFPTest::TestBase<T> {
public:
  xPc1EqC2(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)get_tiny1<T>(),
      (T)get_tiny2<T>()
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] + ti.vals[1] == ti.vals[2];
    return {res?1.0:0.0, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xPc1EqC2)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
//   using namespace CUHelpers;
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx].s1 = ;
//   results[idx].s2 = ;
// }

template <typename T>
class xPc1NeqC2: public QFPTest::TestBase<T> {
public:
  xPc1NeqC2(std::string id) : QFPTest::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual QFPTest::TestInput<T> getDefaultInput(){
    QFPTest::TestInput<T> ti;
    ti.vals = {
      (T)getRandSeq<T>()[0],
      (T)get_tiny1<T>(),
      (T)get_tiny2<T>()
    };
    return ti;
  }
protected:
  virtual
  QFPTest::KernelFunction<T>* getKernel() {return NULL; }
  virtual
  QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    auto res = ti.vals[0] + ti.vals[1] != ti.vals[2];
    return {res?1.0:0.0, 0.0};
  }
  using QFPTest::TestBase<T>::id;
};
REGISTER_TYPE(xPc1NeqC2)
