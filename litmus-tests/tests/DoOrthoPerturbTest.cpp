#include <flit.h>

#include <typeinfo>
#include <iomanip>

#include <cmath>

template <typename T>
GLOBAL
void
DoOPTKernel(const flit::CuTestInput<T>* tiList, flit::CudaResultElement* results){
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif

  auto ti = tiList[idx];
  auto iters = ti.iters;
  auto dim = ti.highestDim;
  double score = 0.0;
  cuvector<unsigned> orthoCount(dim, 0.0);
  // we use a double literal above as a workaround for Intel 15-16 compiler
  // bug:
  // https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
  flit::VectorCU<T> a(ti.vals, ti.length);
  flit::VectorCU<T> b = a.genOrthoVector();

  T backup;

  for(decltype(dim) r = 0; r < dim; ++r){
    T &p = a[r];
    backup = p;
    for(decltype(iters) i = 0; i < iters; ++i){
      auto tmp = flit::as_int(p);
      p = flit::as_float(++tmp); //yeah, this isn't perfect
      //p = std::nextafter(p, std::numeric_limits<T>::max());
      auto watchPoint = FLT_MIN;
      watchPoint = a ^ b;

      bool isOrth = watchPoint == 0; //a.isOrtho(b);
      if(isOrth){
        orthoCount[r]++;
        // score should be perturbed amount
        if(i != 0) score += std::abs(p - backup);
      }else{
        // if falsely not detecting ortho, should be the dot prod
        if(i == 0) score += std::abs(watchPoint); //a ^ b);  
      }
    }
    p = backup;
  }
  results[idx].s1 = score;
  results[idx].s2 = 0;
}

template <typename T>
class DoOrthoPerturbTest : public flit::TestBase<T> {
public:
  DoOrthoPerturbTest(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 16; }
  virtual flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.iters = 200;
    ti.ulp_inc = 1;

    auto dim = getInputsPerRun();
    ti.highestDim = dim;
    ti.vals = std::vector<T>(dim);
    for(decltype(dim) x = 0; x < dim; ++x) ti.vals[x] = static_cast<T>(1 << x);

    return ti;
  }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return DoOPTKernel; }
  virtual
  flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
    using flit::operator<<;

    auto iters = ti.iters;
    auto dim = getInputsPerRun();
    auto ulp_inc = ti.ulp_inc;
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0.0);
    // we use a double literal above as a workaround for Intel 15-16 compiler
    // bug:
    // https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
    flit::Vector<T> a(ti.vals);
    flit::Vector<T> b = a.genOrthoVector();

    T backup;

    for(decltype(dim) r = 0; r < dim; ++r){
      T &p = a[r];
      backup = p;
      for(decltype(iters) i = 0; i < iters; ++i){
        //cout << "r:" << r << ":i:" << i << std::std::endl;

        p = std::nextafter(p, std::numeric_limits<T>::max());
        // Added this for force watchpoint hits every cycle (well, two).  We
        // shouldn't really be hitting float min
        auto watchPoint = FLT_MIN;
        watchPoint = a ^ b;

        // DELME debug
        //std::cout << watchPoint << std::endl;

        bool isOrth = watchPoint == 0; //a.isOrtho(b);
        if(isOrth){
          orthoCount[r]++;
          // score should be perturbed amount
          if(i != 0) score += std::abs(p - backup);
        }else{
          // if falsely not detecting ortho, should be the dot prod
          if(i == 0) score += std::abs(watchPoint); //a ^ b);
        }
        flit::info_stream
          << "i:" << i
          << ":a[" << r << "] = " << a[r] << ", " << flit::as_int(a[r])
          << " multiplier: " << b[r] << ", " << flit::as_int(b[r])
          << " perp: " << isOrth
          << " dot prod: " << flit::as_int(a ^ b)
          << std::endl;
      }
      flit::info_stream << "next dimension . . . " << std::endl;
      p = backup;
    }
    flit::info_stream << "Final report, one iteration set per dimensiion:" << std::endl;
    flit::info_stream << '\t' << "ulp increment per loop: " << ulp_inc << std::endl;
    flit::info_stream << '\t' << "iterations per dimension: " << iters << std::endl;
    flit::info_stream << '\t' << "dimensions: " << dim << std::endl;
    flit::info_stream << '\t' << "precision (type): " << typeid(T).name() << std::endl;
    int cdim = 0;
    for(auto d: orthoCount){
      int exp = 0;
      std::frexp(a[cdim] * b[cdim], &exp);
      flit::info_stream
        << "For mod dim " << cdim << ", there were " << d
        << " ortho vectors, product magnitude (biased fp exp): " << exp
        << std::endl;
      cdim++;
    }
    return {std::pair<long double, long double>(score, 0.0), 0};
  }

private:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoOrthoPerturbTest)
