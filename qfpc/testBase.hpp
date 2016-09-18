// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TEST_BASE_HPP
#define TEST_BASE_HPP

#include "QFPHelpers.hpp"

#ifdef __CUDA__
#include "CUHelpers.hpp"
#endif // __CUDA__

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace QFPTest {

void setWatching(bool watch = true);


using ResultType = std::map<std::pair<const std::string, const std::string>,
                            std::pair<long double, long double>>;


std::ostream&
operator<<(std::ostream&, const ResultType&);

template <typename T>
struct TestInput {
  size_t iters;
  size_t highestDim;
  size_t ulp_inc;
  T min;
  T max;
  std::vector<T> vals;
};

struct CudaResultElement {
  double s1;
  double s2;
};

template <typename T>
struct CuTestInput {
  size_t iters;
  size_t highestDim;
  size_t ulp_inc;
  T min;
  T max;
  cuvector<T> vals;

  // In a separate function instead of constructor to still allow initializer
  // lists to construct
  static CuTestInput<T> fromTestInput(const TestInput<T>& ti) {
    return CuTestInput<T> {
      ti.iters,
      ti.highestDim,
      ti.ulp_inc,
      ti.min,
      ti.max,
      cuvector<T>(ti.vals),
    };
  }
};

template <typename T>
using KernelFunction = void (const CuTestInput<T>*, CudaResultElement*);

/** Calls a CUDA kernel function and returns the scores
 *
 * This function is expecting a non-nullptr kernel function with a test input
 * sufficient to run the test exactly once.
 *
 * If we are not compiling under CUDA, then this does nothing and returns
 * zeros.  If we are compiling under CUDA, then we may run the kernel on the
 * CPU or on the GPU based on the definition of __CPUKERNEL__.
 *
 * This function handles copying the test input to the kernel, allocating
 * memory for the result storage, and copying the result type back to the host
 * to be returned.
 *
 * @param kernel: A kernel function pointer
 * @param cti: The test input with enough inputs in cti.vals for exactly one
 *   run
 */
template <typename T>
std::vector<ResultType::mapped_type>
callKernel(KernelFunction<T>* kernel, const std::vector<TestInput<T>>& tiList) {
#ifdef __CUDA__
  std::unique_ptr<CuTestInput<T>[]> ctiList(new CuTestInput<T>[tiList.size()]);
  for (size_t i = 0; i < tiList.size(); i++) {
    ctiList[i] = CuTestInput<T>::fromTestInput(tiList[i]);
  }
  std::unique_ptr<CudaResultElement[]> cuResults(new CudaResultElement[tiList.size()]);
  // Note: __CPUKERNEL__ mode is broken by the change to run the kernel in
  // multithreaded mode.  Its compilation is broken.
  // TODO: fix __CPUKERNEL__ mode for testing.
 #ifdef __CPUKERNEL__
  kernel(ctiList, cuResults);
 #else  // not __CPUKERNEL__
  CuTestInput<T>* deviceInput;
  CudaResultElement* deviceResult;
  const auto inputSize = sizeof(CuTestInput<T>) * tiList.size();
  const auto resultSize = sizeof(CudaResultElement) * tiList.size();
  checkCudaErrors(cudaMalloc(&deviceInput, inputSize));
  checkCudaErrors(cudaMalloc(&deviceResult, resultSize));
  checkCudaErrors(cudaMemcpy(deviceInput, ctiList.get(), inputSize, cudaMemcpyHostToDevice));
  kernel<<<tiList.size(),1>>>(deviceInput, deviceResult);
  checkCudaErrors(cudaMemcpy(cuResults.get(), deviceResult, resultSize, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(deviceInput));
  checkCudaErrors(cudaFree(deviceResult));
 #endif // __CPUKERNEL__
  std::vector<ResultType::mapped_type> results;
  for (size_t i = 0; i < tiList.size(); i++) {
    results.emplace_back(cuResults[i].s1, cuResults[i].s2);
  }
  return results;
#else  // not __CUDA__
  // Do nothing
  Q_UNUSED(kernel);
  Q_UNUSED(tiList);
  return {};
#endif // __CUDA__
}

template<typename T>
void pushWatchData();

template<typename T>
void popWatchData();

template<typename T>
volatile T& getWatchData();

template <typename T>
class TestBase {
public:
  TestBase(std::string id) : id(std::move(id)) {}
  virtual ~TestBase() = default;

  /** Run the test with the given inputs.
   *
   * This function will need enough inputs for the test otherwise it will throw
   * an exception.  See the documentation for getInputsPerRun().
   *
   * @note This method is virtual, so it can be overridden by the test class if
   *   the test is such that it needs to change the notion of running the test
   *   from only a TestInput object for each result pair.
   *
   * @see getInputsPerRun
   */
  virtual ResultType run(const TestInput<T>& ti) {
    ResultType results;
    auto stride = getInputsPerRun();
    auto runCount = ti.vals.size() / stride;

    // Split up the input.  One for each run
    auto begin = ti.vals.begin();
    std::vector<TestInput<T>> inputSequence;
    TestInput<T> emptyInput {
      ti.iters, ti.highestDim, ti.ulp_inc, ti.min, ti.max, {} };
    emptyInput.vals.clear();
    for (decltype(runCount) i = 0; i < runCount; i++) {
      auto end = begin + stride;
      TestInput<T> testRunInput = emptyInput;
      testRunInput.vals = std::vector<T>(begin, end);
      inputSequence.push_back(testRunInput);
      begin = end;
    }

    // Run the tests
    std::vector<ResultType::mapped_type> scoreList;
#ifdef __CUDA__
    ResultType::mapped_type scores;
    auto kernel = getKernel();
    if (kernel == nullptr) {
      for (auto runInput : inputSequence) {
        scoreList.push_back(run_impl(runInput));
      }
    } else {
      scoreList = callKernel(kernel, inputSequence);
    }
#else  // not __CUDA__
    for (auto runInput : inputSequence) {
      scoreList.push_back(run_impl(runInput));
    }
#endif // __CUDA__

    // Store and return the test results
    for (size_t i = 0; i < scoreList.size(); i++) {
      std::string name = id;
      if (runCount != 1) {
        name += "_idx" + std::to_string(i);
      }
      results.insert({{name, typeid(T).name()}, scoreList[i]});
    }
    return results;
  }

  /** This is a set of default inputs to use for the test
   *
   * This function should be implemented such that we can simply call this test
   * in the following way:
   *   test->run(test->getDefaultInput());
   */
  virtual TestInput<T> getDefaultInput() = 0;

  /** The number of inputs per test run
   *
   * If the test input to the public run() function has less than this value,
   * then an exception will be thrown from run().
   *
   * If the test input has many more inputs than this, then the run() function
   * will run the test as many times as it can with the amount of values given.
   * For example, if the test requires 5 inputs, and the TestInput object
   * passed in has 27 inputs, then the test will be run 5 times, and the last
   * two inputs will be unused.
   */
  virtual size_t getInputsPerRun() = 0;

protected:
  /** If this test implements a CUDA kernel, return the kernel pointer
   *
   * If compiling under CUDA and this returns a valid function pointer (meaning
   * not nullptr), then this kernel function will be called instead of
   * run_impl().  Otherwise, run_impl() will be called.
   *
   * This method is not pure virtual because it is not required to implement.
   * If it is overridden to return something other than nullptr, then the
   * kernel will be called when compiling under a CUDA environment.  If when
   * compiling under a CUDA environment, this returns nullptr, then the test
   * reverts to calling run().
   */
  virtual KernelFunction<T>* getKernel() { return nullptr; }

  /** This is where you implement the test
   *
   * @param ti: Test input.  The vals element will have exactly the amount of
   *   test inputs required according to the implemented getInputsPerRun().  So
   *   if that function returns 9, then the vector will have exactly 9
   *   elements.
   * @return a single result.  See ResultType to see what the mapped types is.
   */
  virtual ResultType::mapped_type run_impl(const TestInput<T>& ti) = 0;

protected:
  const std::string id;
};

/// A completely empty test that outputs nothing
template <typename T>
class NullTest : public TestBase<T> {
public:
  NullTest(std::string id) : TestBase<T>(std::move(id)) {}
  virtual TestInput<T> getDefaultInput() { return {}; }
  virtual size_t getInputsPerRun() { return 0; }
  virtual ResultType run(const TestInput<T>&) { return {}; }
protected:
  virtual ResultType::mapped_type run_impl(const TestInput<T>&) { return {}; }
};

class TestFactory {
public:
  template <typename F> std::shared_ptr<TestBase<F>> get();

protected:
  using createType =
    std::tuple<
      std::shared_ptr<TestBase<float>>,
      std::shared_ptr<TestBase<double>>,
      std::shared_ptr<TestBase<long double>>
      >;
  virtual createType create() = 0;

private:
  void makeSureIsCreated() {
    // If the shared_ptr is empty, then create the tests
    if (std::get<0>(_tests).use_count() == 0) {
      _tests = create();
    }
  }

private:
  createType _tests;
};

template <>
inline std::shared_ptr<TestBase<float>> TestFactory::get<float> () {
  makeSureIsCreated();
  return std::get<0>(_tests);
}

template <>
inline std::shared_ptr<TestBase<double>> TestFactory::get<double> () {
  makeSureIsCreated();
  return std::get<1>(_tests);
}

template <>
inline std::shared_ptr<TestBase<long double>> TestFactory::get<long double> () {
  makeSureIsCreated();
  return std::get<2>(_tests);
}

#ifdef __CUDA__

#define REGISTER_TYPE(klass)                                \
  class klass##Factory : public QFPTest::TestFactory {      \
  public:                                                   \
    klass##Factory() {                                      \
      QFPTest::registerTest(#klass, this);                  \
    }                                                       \
  protected:                                                \
    virtual createType create() {                           \
      return std::make_tuple(                               \
          std::make_shared<klass<float>>(#klass),           \
          std::make_shared<klass<double>>(#klass),          \
          /* empty test for long double */                  \
          std::make_shared<QFPTest::NullTest<long double>>(#klass) \
          );                                                \
    }                                                       \
  };                                                        \
  static klass##Factory global_##klass##Factory;            \

#else // not __CUDA__

#define REGISTER_TYPE(klass)                                \
  class klass##Factory : public QFPTest::TestFactory {      \
  public:                                                   \
    klass##Factory() {                                      \
      QFPTest::registerTest(#klass, this);                  \
    }                                                       \
  protected:                                                \
    virtual createType create() {                           \
      return std::make_tuple(                               \
          std::make_shared<klass<float>>(#klass),           \
          std::make_shared<klass<double>>(#klass),          \
          std::make_shared<klass<long double>>(#klass)      \
          );                                                \
    }                                                       \
  };                                                        \
  static klass##Factory global_##klass##Factory;            \

#endif // __CUDA__

inline std::map<std::string, TestFactory*>& getTests() {
  static std::map<std::string, TestFactory*> tests;
  return tests;
}

// template <bool C = hasCuda, typename std::enable_if<!C>::type* = nullptr>
// static std::map<std::string, TestFactory*>& getTests() {
// #ifdef __CUDA__
//   return {};
// #else
//   static std::map<std::string, TestFactory*> tests;
//   return tests;
// #endif
// }

inline void registerTest(const std::string& name, TestFactory *factory) {
  getTests()[name] = factory;
}

}


/** A convenience macro for running Eigen tests
 *
 * @internal The run() method is overridden so that the methods
 *   getDefaultInput(), getInputsPerRun() and run_impl() are not necessary, so
 *   the minimum implementation is present here.
 */
#define EIGEN_CLASS_DEF(klass, file)                     \
  template <typename T>                                  \
  class klass : public QFPTest::TestBase<T> {            \
  public:                                                \
    klass(std::string id)                                \
      : QFPTest::TestBase<T>(std::move(id)) {}           \
    virtual QFPTest::ResultType                          \
    run(const QFPTest::TestInput<T>& ti) {               \
      Q_UNUSED(ti);                                      \
      if(sizeof(T) != 4) return {};                      \
      auto fileS = std::string(#file);                   \
      g_test_stack[fileS];                               \
      eigenResults[fileS];                               \
      test_##file();                                     \
      g_test_stack[fileS].clear();                       \
      auto res = eigenResults[fileS];                    \
      eigenResults[fileS].clear();                       \
      return res;                                        \
    }                                                    \
    virtual QFPTest::TestInput<T> getDefaultInput() {    \
      QFPTest::TestInput<T> ti;                          \
      return ti;                                         \
    }                                                    \
    virtual size_t getInputsPerRun() { return 0; }       \
  protected:                                             \
    virtual QFPTest::KernelFunction<T>*                  \
    getKernel() {return nullptr; }                       \
    virtual QFPTest::ResultType::mapped_type             \
    run_impl(const QFPTest::TestInput<T>&) { return {}; }\
  };                                                     \


#endif // TEST_BASE_HPP
