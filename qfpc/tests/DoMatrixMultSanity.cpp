#include "testBase.hpp"
#include "QFPHelpers.hpp"

#include <cmath>
#include <typeinfo>

using QFPHelpers::Matrix;
using QFPHelpers::Vector;
using QFPHelpers::info_stream;
using QFPTest::ResultType;
using QFPTest::TestBase;
using QFPTest::TestInput;

template <typename T>
class DoMatrixMultSanity: public TestBase<T> {
public:
  DoMatrixMultSanity(std::string id) : TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 16; }

  virtual TestInput<T> getDefaultInput() {
    TestInput<T> ti;
    ti.highestDim = getInputsPerRun();
    ti.min = -6;
    ti.max = 6;
    ti.vals = Vector<T>::getRandomVector(ti.highestDim, ti.min, ti.max).getData();
    return ti;
  }

protected:
  ResultType::mapped_type run_impl(const TestInput<T>& ti) {
    auto dim = getInputsPerRun();
    Vector<T> b = {ti.vals[0], ti.vals[1], ti.vals[2]};
    auto c = Matrix<T>::Identity(dim) * b;
    bool eq = (c == b);
    info_stream << id << ": Product is: " << c << std::endl;
    info_stream << id << ": A * b == b? " << eq << std::endl;
    return {c.L1Distance(b), c.LInfDistance(b)};
  }

protected:
  using TestBase<T>::id;
};

REGISTER_TYPE(DoMatrixMultSanity)
