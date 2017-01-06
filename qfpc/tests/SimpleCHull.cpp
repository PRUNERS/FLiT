#include "testBase.hpp"
#include "QFPHelpers.hpp"

#define WFT float
#include "S3FP/examples/tests-div-detection/simple_convex_hull/simple_convex_hull.cpp"

#include <cmath>
#include <cstdio>
#include <typeinfo>

using namespace CUHelpers;

template <typename T>
class SimpleCHull: public QFPTest::TestBase<T> {
public:
  SimpleCHull(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}
  QFPTest::TestInput<T> getDefaultInput(){return QFPTest::TestInput<T>();}
  QFPTest::ResultType run(const QFPTest::TestInput<T>& ti) {
    QFPTest::ResultType results;
    std::vector<QFPTest::ResultType::mapped_type> scoreList;
    results.insert({{id, typeid(T).name()}, run_impl(ti)});
    return results;
  }
  size_t getInputsPerRun(){ return 1;}
protected:
  virtual QFPTest::ResultType::mapped_type
  run_impl(const QFPTest::TestInput<T>& ti) {
    Q_UNUSED(ti);
    CHullEdges.clear();
    PointList.clear();
    ReadInputs(fopen("S3FP/examples/tests-div-detection/simple_convex_hull/random_input", "r"));
    SimpleComputeConvexhull<T>();
    return {(long double) getEdgeCount(), 0.0};
  }

protected:
  using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(SimpleCHull)
