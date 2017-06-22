#include "simple_convex_hull.h"

#include <flit.h>

#include <cmath>
#include <cstdio>
#include <typeinfo>

template <typename T>
class SimpleCHull: public flit::TestBase<T> {
public:
  SimpleCHull(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun(){ return 0; }
  virtual flit::TestInput<T> getDefaultInput(){ return {}; }

protected:
  virtual flit::KernelFunction<T>* getKernel() { return nullptr; }

  virtual flit::ResultType::mapped_type
  run_impl(const flit::TestInput<T>& ti) {
    Q_UNUSED(ti);
    CHullEdges.clear();
    PointList.clear();
    ReadInputs(fopen("data/random_input", "r"));
    SimpleComputeConvexhull<T>();
    return {std::pair<long double, long double>((long double)
						getEdgeCount(), 0.0), 0};
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(SimpleCHull)
