#include <flit.h>

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <type_traits>

namespace {

  template <typename T>
  T get_tiny1() { static_assert(false, "Unimplemented type"); }

  template <typename T>
  T get_tiny2() { static_assert(false, "Unimplemented type"); }

  template <>
  float get_tiny1<float>(){
    return 1.175494351-38;
  }

  template <>
  double get_tiny1<double>(){
    return 2.2250738585072014e-308;
  }

  template <>
  long double get_tiny1<long double>(){
    return 3.362103143112093506262e-4931L;
  }

  template <>
  float get_tiny2<float>(){
    return 1.175494352-38;
  }

  template <>
  double get_tiny2<double>(){
    return 2.2250738585072015e-308;
  }

  template <>
  long double get_tiny2<long double>(){
    return 3.362103143112093506263e-4931L;
  }

} // end of unnamed namespace

template <typename T>
class FtoDecToF: public flit::TestBase<T> {
public:
  FtoDecToF(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return {std::nextafter(T(0.0), T(1.0))};
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
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

template <typename T>
class subnormal: public flit::TestBase<T> {
public:
  subnormal(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return {std::nextafter(T(0.0), T(1.0))};
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    return ti.vals[0] - ti.vals[0] / 2;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(subnormal)

template <typename T>
class dotProd: public flit::TestBase<T> {
public:
  dotProd(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
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

template <typename T>
class simpleReduction: public flit::TestBase<T> {
public:
  simpleReduction(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
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

template <typename T>
class addTOL : public flit::TestBase<T> {
public:
  addTOL(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(man_bits + 1, nls.max_exponent);
    // generate the range of offsets, and then generate the
    // mantissa bits for each of the three inputs
    auto L1e = dis(gen); //L1 exponent

    // for the ldexp function we're using, it takes an unbiased exponent and
    // there is no implied 1 MSB for the mantissa / significand
    T zero = 0.0;
    auto L1m = flit::as_int(zero);
    auto L2m = flit::as_int(zero);
    auto sm = flit::as_int(zero);
    for(int i = 0; i < man_bits; ++i){
      L1m &= (gen() & 1) << i;
      L2m &= (gen() & 1) << i;
      sm  &= (gen() & 1) << i;
    }
    return {
      std::ldexp(flit::as_float(L1m), L1e),
      std::ldexp(flit::as_float(L2m), L1e - 1),
      std::ldexp(flit::as_float(sm), L1e - man_bits)
    };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] + ti[1] + ti[2];
    return res;
  }

  using flit::TestBase<T>::id;
};
REGISTER_TYPE(addTOL)

template <typename T>
class addSub: public flit::TestBase<T> {
public:
  addSub(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { T(1.0) };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    auto big = std::pow(2, (T)man_bits - 1);
    auto res = (ti[0] + big) - big;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(addSub)

template <typename T>
class divc: public flit::TestBase<T> {
public:
  divc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] / ti[1];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(divc)

template <typename T>
class zeroMinusX: public flit::TestBase<T> {
public:
  zeroMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = T(0.0) - ti[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroMinusX)

template <typename T>
class xMinusZero: public flit::TestBase<T> {
public:
  xMinusZero(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] - T(0.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusZero)

template <typename T>
class zeroDivX: public flit::TestBase<T> {
public:
  zeroDivX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = (T)0.0 / ti.vals[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroDivX)

template <typename T>
class xDivOne: public flit::TestBase<T> {
public:
  xDivOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T res = ti[0] / T(1.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivOne)

template <typename T>
class xDivNegOne: public flit::TestBase<T> {
public:
  xDivNegOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T res = ti[0] / T(-1.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivNegOne)

template <typename T>
class negAdivB: public flit::TestBase<T> {
public:
  negAdivB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] / ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAdivB)

template <typename T>
class negAminB: public flit::TestBase<T> {
public:
  negAminB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] - ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAminB)

template <typename T>
class xMinusX: public flit::TestBase<T> {
public:
  xMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { flit::getRandSeq<T>()[0] };
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] - ti[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusX)

template <typename T>
class negAplusB: public flit::TestBase<T> {
public:
  negAplusB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] + ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAplusB)

template <typename T>
class aXbDivC: public flit::TestBase<T> {
public:
  aXbDivC(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] * (ti[1] / ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbDivC)

template <typename T>
class aXbXc: public flit::TestBase<T> {
public:
  aXbXc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] * (ti[1] * ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbXc)

template <typename T>
class aPbPc: public flit::TestBase<T> {
public:
  aPbPc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      flit::getRandSeq<T>()[1],
      flit::getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] + (ti[1] + ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aPbPc)

template <typename T>
class xPc1EqC2: public flit::TestBase<T> {
public:
  xPc1EqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      get_tiny1<T>(),
      get_tiny2<T>(),
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    bool res = ti[0] + ti[1] == ti[2];
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1EqC2)

template <typename T>
class xPc1NeqC2: public flit::TestBase<T> {
public:
  xPc1NeqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      flit::getRandSeq<T>()[0],
      get_tiny1<T>(),
      get_tiny2<T>(),
    };
  }
protected:
  virtual flit::KernelFunction<T>* getKernel() override { return nullptr; }

  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    bool res = (ti[0] + ti[1] != ti[2]);
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1NeqC2)
