#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>
#include <iomanip>

template <typename T>
class DoOrthoPerturbTest: public QFPTest::TestBase {
public:
  DoOrthoPerturbTest(std::string id):QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
#ifdef __CUDA__
    return {{{id, typeid(T).name()}, {0.0, 0.0}}};
#else
    using namespace QFPHelpers;
    using namespace QFPHelpers::FPHelpers;
    auto iters = ti.iters;
    auto dim = ti.highestDim;
    size_t indexer = 0;
    auto ulp_inc = ti.ulp_inc;
    auto fun = [&indexer](){return (T)(1 << indexer++);};
    //    auto fun = [&indexer](){return 2.0 / pow((T)10.0, indexer++);};
    auto& watchPoint = QFPTest::getWatchData<T>();
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0.0);
    // we use a double literal above as a workaround for Intel 15-16 compiler
    // bug:
    // https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
    QFPHelpers::Vector<T> a(dim, fun);
    QFPHelpers::Vector<T> b = a.genOrthoVector();

    QFPHelpers::info_stream << "starting dot product orthogonality test with a, b = "
                            << std::endl;
    for(decltype(dim) x = 0; x < dim; ++x)
      QFPHelpers::info_stream << x << '\t';
    QFPHelpers::info_stream << std::endl;
    QFPHelpers::info_stream << a << std::endl;
    QFPHelpers::info_stream << b << std::endl;
    T backup;

    for(decltype(dim) r = 0; r < dim; ++r){
    T &p = a[r];
      backup = p;
      for(decltype(iters) i = 0; i < iters; ++i){
        //cout << "r:" << r << ":i:" << i << std::std::endl;
        //        p = QFPHelpers::FPHelpers::perturbFP(backup, i * ulp_inc);
        
        p = std::nextafter(p, std::numeric_limits<T>::max());
        // Added this for force watchpoint hits every cycle (well, two).  We
        // shouldn't really be hitting float min
        watchPoint = FLT_MIN;
        watchPoint = a ^ b;

        // DELME debug
        //std::cout << watchPoint << std::endl;

        bool isOrth = watchPoint == 0; //a.isOrtho(b);
        if(isOrth){
          orthoCount[r]++;
          // score should be perturbed amount
          if(i != 0) score += fabs(p - backup);
        }else{
          // if falsely not detecting ortho, should be the dot prod
          if(i == 0) score += fabs(watchPoint); //a ^ b);  
        }
        QFPHelpers::info_stream << "i:" << i << ":a[" << r << "] = " <<
          a[r] << ", " << as_int(a[r]) << " multiplier: " <<
          b[r] << ", " << as_int(b[r]) <<
          " perp: " << isOrth << " dot prod: " <<
          as_int(a ^ b) << std::endl;
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
      int exp = 0;
      std::frexp(a[cdim] * b[cdim], &exp);
      QFPHelpers::info_stream << "For mod dim " << cdim <<
        ", there were " << d <<
        " ortho vectors, product magnitude (biased fp exp): " <<
        exp << std::endl;
      cdim++;
    }
    return {{{id, typeid(T).name()}, {score, 0.0}}};
#endif
  }
};

REGISTER_TYPE(DoOrthoPerturbTest)
