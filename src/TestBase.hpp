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
#include <cassert>

namespace QFPTest {

void setWatching(bool watch = true);


using ResultType = std::map<std::pair<const std::string, const std::string>,
                            std::pair<std::pair<long double, long double>, int_fast64_t>>;

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

/** A simple structure used in CUDA tests.
 *
 * It stores some values and a pointer, but does not do dynamic allocation nor
 * deallocation.  The struct does not own the vals pointer at all, just holds
 * its value.
 *
 * The vals are intended to be read-only, because they are inputs.
 */
template <typename T>
struct CuTestInput {
  size_t iters = 100;
  size_t highestDim = 0;
  size_t ulp_inc = 1;
  T min = -6;
  T max = 6;
  const T* vals = nullptr; // values with size length
  size_t length = 0;

  /** Creates a CuTestInput object containing the same info as the TestInput
   *
   * This is in a separate function instead of constructor to still allow
   * initializer lists to construct
   *
   * Note, the vals pointer will point to the internal data from the TestInput
   * object.  It is unsafe for the CuTestInput object to outlive this TestInput
   * object unless you set a new value for vals.
   */
  static CuTestInput<T> fromTestInput(const TestInput<T>& ti) {
    CuTestInput<T> cti {};
    cti.iters       = ti.iters;
    cti.highestDim  = ti.highestDim;
    cti.ulp_inc     = ti.ulp_inc;
    cti.min         = ti.min;
    cti.max         = ti.max;
    cti.vals        = ti.vals.data();
    cti.length      = ti.vals.size();
    return cti;
  }
};

/** Definition of a kernel function used by CUDA tests
 *
 * @param arr: array of test input objects, already allocated and populated
 * @param results: array where to store results, already allocated
 */
template <typename T>
using KernelFunction = void (const CuTestInput<T>*, CudaResultElement*);

template <typename T>
using CudaDeleter = void (T*);

template <typename T>
std::unique_ptr<T, CudaDeleter<T>*> makeCudaArr(const T* vals, size_t length) {
#ifdef __CUDA__
  T* arr;
  const auto arrSize = sizeof(T) * length;

  // Create the array
  checkCudaErrors(cudaMalloc(&arr, arrSize));

  // Store it in a smart pointer with a custom deleter
  CudaDeleter<T>* deleter = [](T* p) { checkCudaErrors(cudaFree(p)); };
  std::unique_ptr<T, CudaDeleter<T>*> ptr(arr, deleter);

  // Copy over the vals array from hist into the device
  if (vals != nullptr) {
    checkCudaErrors(cudaMemcpy(ptr.get(), vals, arrSize, cudaMemcpyHostToDevice));
  }

  return ptr;
#else
  Q_UNUSED(vals);
  Q_UNUSED(length);
  throw std::runtime_error("Should not use makeCudaArr without CUDA enabled");
#endif
}

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
 * @param kernel: kernel function pointer to call with split up inputs
 * @param ti: inputs for all tests runs, to be split by stride
 * @param stride: how many inputs per test run
 */
template <typename T>
std::vector<ResultType::mapped_type>
runKernel(KernelFunction<T>* kernel, const TestInput<T>& ti, size_t stride) {
#ifdef __CUDA__
  size_t runCount;
  if (stride < 1) { // the test takes no inputs
    runCount = 1;
  } else {
    runCount = ti.vals.size() / stride;
  }

  std::unique_ptr<CuTestInput<T>[]> ctiList(new CuTestInput<T>[runCount]);
  for (size_t i = 0; i < runCount; i++) {
    ctiList[i] = CuTestInput<T>::fromTestInput(ti);
    // just point to a place in the array, like a slice
    ctiList[i].vals = ti.vals.data() + i * stride;
    ctiList[i].length = stride;
  }
  std::unique_ptr<CudaResultElement[]> cuResults(new CudaResultElement[runCount]);
  // Note: __CPUKERNEL__ mode is broken by the change to run the kernel in
  // multithreaded mode.  Its compilation is broken.
  // TODO: fix __CPUKERNEL__ mode for testing.
# ifdef __CPUKERNEL__
  kernel(ctiList, cuResults);
# else  // not __CPUKERNEL__
  auto deviceVals = makeCudaArr(ti.vals.data(), ti.vals.size());
  // Reset the pointer value to device addresses
  for (size_t i = 0; i < runCount; i++) {
    ctiList[i].vals = deviceVals.get() + i * stride;
  }
  auto deviceInput = makeCudaArr(ctiList.get(), runCount);
  auto deviceResult = makeCudaArr<CudaResultElement>(nullptr, runCount);
  kernel<<<runCount,1>>>(deviceInput.get(), deviceResult.get());
  auto resultSize = sizeof(CudaResultElement) * runCount;
  checkCudaErrors(cudaMemcpy(cuResults.get(), deviceResult.get(), resultSize,
                             cudaMemcpyDeviceToHost));
# endif // __CPUKERNEL__
  std::vector<ResultType::mapped_type> results;
  for (size_t i = 0; i < runCount; i++) {
    results.emplace_back(std::pair<long double, long double>
                         (cuResults[i].s1, cuResults[i].s2), 0);
  }
  return results;
#else  // not __CUDA__
  // Do nothing
  Q_UNUSED(kernel);
  Q_UNUSED(ti);
  Q_UNUSED(stride);
  return {};
#endif // __CUDA__
}

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
  virtual ResultType run(const TestInput<T>& ti,
                         const bool GetTime,
                         const size_t TimingLoops) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    ResultType results;
    TestInput<T> emptyInput {
      ti.iters, ti.highestDim, ti.ulp_inc, ti.min, ti.max, {}
    };
    auto stride = getInputsPerRun();
    std::vector<TestInput<T>> inputSequence;

    if (stride < 1) { // the test does not take any inputs
      inputSequence.push_back(ti);
    } else {
      // Split up the input.  One for each run
      auto begin = ti.vals.begin();
      auto runCount = ti.vals.size() / stride;
      for (decltype(runCount) i = 0; i < runCount; i++) {
        auto end = begin + stride;
        TestInput<T> testRunInput = emptyInput;
        testRunInput.vals = std::vector<T>(begin, end);
        inputSequence.push_back(testRunInput);
        begin = end;
      }
    }

    // Run the tests
    std::vector<ResultType::mapped_type> scoreList;
#ifdef __CUDA__
    auto kernel = getKernel();
    if (kernel == nullptr) {
      for (auto runInput : inputSequence) {
        if (GetTime) {
          ResultType::mapped_type scores;
          int_fast64_t nsecs = 0;
          for (int r = 0; r < TimingLoops; ++r) {
            auto s = high_resolution_clock::now();
            scores = run_impl(runInput);
            auto e = high_resolution_clock::now();
            nsecs += duration_cast<duration<int_fast64_t, std::nano>>(e-s).count();
            assert(nsecs > 0);
          }
          scores.second = nsecs / TimingLoops;
          scoreList.push_back(scores);
        } else {
          scoreList.push_back(run_impl(runInput));
        }
      }
    } else {
      if (GetTime) {
        ResultType::mapped_type scores;
        int_fast64_t nsecs = 0;
        for (size_t r = 0; r < TimingLoops; ++r){
          auto s = high_resolution_clock::now();
          scoreList = runKernel(kernel, ti, stride);
          auto e = high_resolution_clock::now();
          nsecs += duration_cast<duration<int_fast64_t, std::nano>>(e-s).count();
          assert(nsecs > 0);
        }
        auto avg = nsecs / TimingLoops;
        auto avgPerKernel = avg / scoreList.size();
        for (auto& s : scoreList) {
          s.second = avgPerKernel;
        }
      } else {
        scoreList = runKernel(kernel, ti, stride);
      }
    }
#else  // not __CUDA__
    for (auto runInput : inputSequence) {
      if (GetTime) {
        ResultType::mapped_type scores;
        int_fast64_t nsecs = 0;
        for (size_t r = 0; r < TimingLoops; ++r) {
          auto s = high_resolution_clock::now();
          scores = run_impl(runInput);
          auto e = high_resolution_clock::now();
          nsecs += duration_cast<duration<int_fast64_t, std::nano>>(e-s).count();
          assert(nsecs > 0);
        }
        scores.second = nsecs / TimingLoops;
        scoreList.push_back(scores);
      } else {
        scoreList.push_back(run_impl(runInput));
      }
    }
#endif // __CUDA__

    // Store and return the test results
    for (size_t i = 0; i < scoreList.size(); i++) {
      std::string name = id;
      if (scoreList.size() != 1) {
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
  virtual ResultType run(const TestInput<T>&,
                         const bool,
                         const bool) { return {}; }
protected:
  virtual KernelFunction<T>* getKernel() { return nullptr; }
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

} // end namespace QFPTest

#endif // TEST_BASE_HPP
