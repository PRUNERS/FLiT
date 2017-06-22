#include <flit.h>

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <type_traits>


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class FtoDecToF: public flit::TestBase<T> {
public:
  FtoDecToF(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput() {
  flit::TestInput<T> ti;
    ti.vals = {std::nextafter(T(0.0), T(1.0))};
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    std::numeric_limits<T> nlim;
    // from https://en.wikipedia.org/wiki/IEEE_floating_point
    uint16_t ddigs = nlim.digits * std::log10(2) + 1;
    std::ostringstream res;
    res << std::setprecision(ddigs) << ti.vals[0];
    std::string dstr;
    dstr = res.str();
    T backAgain;
    std::istringstream(dstr) >> backAgain;
    return ti.vals[0] - backAgain;
  }

  using flit::TestBase<T>::id;
};

REGISTER_TYPE(FtoDecToF)

// template <typename T>
// GLOBAL
// void
// subnormalKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class subnormal: public flit::TestBase<T> {
public:
  subnormal(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {std::nextafter(T(0.0), T(1.0))};
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    return ti.vals[0] - ti.vals[0] / 2;
  }
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(subnormal)

// template <typename T>
// GLOBAL
// void
// dotProdKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class dotProd: public flit::TestBase<T> {
public:
  dotProd(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 0; }
  virtual flit::TestInput<T> getDefaultInput() { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    auto size = 16;

    auto rand = flit::getRandSeq<T>();

    flit::Vector<T> A(std::vector<T>(rand.begin(),
			    rand.begin() + size));
    flit::Vector<T> B(std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size));
    return A ^ B;
  }

protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(dotProd)

// template <typename T>
// GLOBAL
// void
// simpleReductionKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class simpleReduction: public flit::TestBase<T> {
public:
  simpleReduction(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 0; }
  virtual flit::TestInput<T> getDefaultInput() { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    auto vals = flit::getRandSeq<T>();
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
    return sum;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(simpleReduction)

//This test adds L1 + L2 + s, where L1 & L2 are large, s small

// template <typename T>
// GLOBAL
// void
// addTOLKernel(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class addTOL : public flit::TestBase<T> {
public:
  addTOL(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(man_bits + 1, nls.max_exponent);
    //generate the range of offsets, and then generate the
    //mantissa bits for each of the three inputs
    auto L1e = dis(gen); //L1 exponent

    //for the ldexp function we're using, it takes an unbiased exponent and
    //there is no implied 1 MSB for the mantissa / significand
    T zero = 0.0;
    auto L1m = flit::as_int(zero);
    auto L2m = flit::as_int(zero);
    auto sm = flit::as_int(zero);
    for(int i = 0; i < man_bits; ++i){
      L1m &= (gen() & 1) << i;
      L2m &= (gen() & 1) << i;
      sm  &= (gen() & 1) << i;
    }
    ti.vals = {
      std::ldexp(flit::as_float(L1m), L1e),
      std::ldexp(flit::as_float(L2m), L1e - 1),
      std::ldexp(flit::as_float(sm), L1e - man_bits)
    };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] + ti.vals[1] + ti.vals[2];
    return res;
  }

  using flit::TestBase<T>::id;
};

//the basic idea of this test is A(I) + B + TOL, where A & B are large,
// and TOL is tiny.  
REGISTER_TYPE(addTOL)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class addSub: public flit::TestBase<T> {
public:
  addSub(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = {T(1.0)};
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    auto big = std::pow(2, (T)man_bits - 1);
    auto res = (ti.vals[0] + big) - big;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(addSub)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class divc: public flit::TestBase<T> {
public:
  divc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] / ti.vals[1];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(divc)
// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class zeroMinusX: public flit::TestBase<T> {
public:
  zeroMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = T(0.0) - ti.vals[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroMinusX)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xMinusZero: public flit::TestBase<T> {
public:
  xMinusZero(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] - (T)0.0;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusZero)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class zeroDivX: public flit::TestBase<T> {
public:
  zeroDivX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = (T)0.0 / ti.vals[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroDivX)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xDivOne: public flit::TestBase<T> {
public:
  xDivOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] / (T)1.0;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivOne)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xDivNegOne: public flit::TestBase<T> {
public:
  xDivNegOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] / (T)-1.0;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivNegOne)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class negAdivB: public flit::TestBase<T> {
public:
  negAdivB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = -(ti.vals[0] / ti.vals[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAdivB)

// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

// template <typename T>
// class twiceCast: public flit::TestBase<T> {
// public:
//   twiceCast(std::string id) : flit::TestBase<T>(std::move(id)){}

//   virtual size_t getInputsPerRun() { return 1; }
//   virtual flit::TestInput<T> getDefaultInput(){
//     flit::TestInput<T> ti;
//     ti.vals = { flit::getRandSeq<T>()[0] };
//     return ti;
//   }
// protected:
//   virtual flit::KernelFunction<T>* getKernel() { return nullptr; }
// 
//   virtual long double run_impl(const flit::TestInput<T>& ti) {
//     //yes, this is ugly.  ti.vals s/b vector of floats
//     auto res = (T)((std::result_of<::get_next_type(T)>::type)ti.vals[0]);
//     return res;
//   }
//   using flit::TestBase<T>::id;
// };
// REGISTER_TYPE(twiceCast)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class negAminB: public flit::TestBase<T> {
public:
  negAminB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = -(ti.vals[0] - ti.vals[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAminB)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xMinusX: public flit::TestBase<T> {
public:
  xMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 1; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = { flit::getRandSeq<T>()[0] };
    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] - ti.vals[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusX)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class negAplusB: public flit::TestBase<T> {
public:
  negAplusB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 2; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = -(ti.vals[0] + ti.vals[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAplusB)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class aXbDivC: public flit::TestBase<T> {
public:
  aXbDivC(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] * (ti.vals[1] / ti.vals[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbDivC)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class aXbXc: public flit::TestBase<T> {
public:
  aXbXc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] * (ti.vals[1] * ti.vals[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbXc)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class aPbPc: public flit::TestBase<T> {
public:
  aPbPc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    auto res = ti.vals[0] + (ti.vals[1] + ti.vals[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aPbPc)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xPc1EqC2: public flit::TestBase<T> {
public:
  xPc1EqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::get_tiny1<T>(),
      flit::get_tiny2<T>(),
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    bool res = ti.vals[0] + ti.vals[1] == ti.vals[2];
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1EqC2)


// template <typename T>
// GLOBAL
// void
// FtoDecToFKern(const flit::CuTestInput<T>* tiList, double* results){
// #ifdef __CUDA__
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
// #else
//   auto idx = 0;
// #endif
//   results[idx] = 0.0;
// }

template <typename T>
class xPc1NeqC2: public flit::TestBase<T> {
public:
  xPc1NeqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() { return 3; }
  virtual flit::TestInput<T> getDefaultInput(){
    flit::TestInput<T> ti;
    ti.vals = {
      flit::getRandSeq<T>()[0],
      flit::get_tiny1<T>(),
      flit::get_tiny2<T>(),
    };
    return ti;
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    bool res = ti.vals[0] + ti.vals[1] != ti.vals[2];
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1NeqC2)
