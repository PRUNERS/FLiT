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

  virtual long double run_impl(const flit::TestInput<T>& ti) {
    FLIT_UNUSED(ti);
    CHullEdges.clear();
    PointList.clear();
    ReadInputs(fopen("data/random_input", "r"));
    SimpleComputeConvexhull<T>();
    return getEdgeCount();
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(SimpleCHull)
