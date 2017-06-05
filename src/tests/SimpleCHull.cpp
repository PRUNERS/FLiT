#include "TestBase.hpp"
#include "QFPHelpers.hpp"

//#define SCH_LIB
#define WFT float
#include "S3FP/simple_convex_hull.cpp"

#include <cmath>
#include <cstdio>
#include <typeinfo>

using namespace CUHelpers;

template <typename T>
class SimpleCHull: public QFPTest::TestBase<T> {
public:
  SimpleCHull(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun(){ return 0; }
  virtual QFPTest::TestInput<T> getDefaultInput(){ return {}; }

protected:
  virtual QFPTest::KernelFunction<T>* getKernel() { return nullptr; }

  virtual QFPTest::ResultType::mapped_type
  run_impl(const QFPTest::TestInput<T>& ti) {
    Q_UNUSED(ti);
    CHullEdges.clear();
    PointList.clear();
    ReadInputs(fopen("data/random_input", "r"));
    SimpleComputeConvexhull<T>();
    return {std::pair<long double, long double>((long double)
						getEdgeCount(), 0.0), 0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(SimpleCHull)
