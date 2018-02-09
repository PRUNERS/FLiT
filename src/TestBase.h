/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TEST_BASE_HPP
#define TEST_BASE_HPP

#include "flitHelpers.h"

#ifdef __CUDA__
#include "CUHelpers.h"
#endif // __CUDA__

#include "timeFunction.h"
#include "Variant.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace flit {

void setWatching(bool watch = true);

struct TestResult {
public:
  TestResult(const std::string &_name, const std::string &_precision,
             const Variant &_result, int_fast64_t _nanosecs,
             const std::string &_resultfile = "")
    : m_name(_name)
    , m_precision(_precision)
    , m_result(_result)
    , m_nanosecs(_nanosecs)
    , m_resultfile(_resultfile)
  { }

  // getters
  std::string name() const { return m_name; }
  std::string precision() const { return m_precision; }
  Variant result() const { return m_result; }
  int_fast64_t nanosecs() const { return m_nanosecs; }
  long double comparison() const { return m_comparison; }
  bool is_comparison_null() const { return m_is_comparison_null; }
  std::string resultfile() const { return m_resultfile; }

  // setters
  void set_comparison(long double _comparison) {
    m_comparison = _comparison;
    m_is_comparison_null = false;
  }
  void set_resultfile(const std::string &_resultfile) {
    m_resultfile = _resultfile;
  }

private:
  std::string m_name;
  std::string m_precision;
  Variant m_result;
  int_fast64_t m_nanosecs {0};
  long double m_comparison {0.0L};
  bool m_is_comparison_null {true};
  std::string m_resultfile;
};

std::ostream& operator<<(std::ostream& os, const TestResult& res);

/** Definition of a kernel function used by CUDA tests
 *
 * @param arr: array of input arrays, flattened, already allocated and populated
 * @param n: length of each input, it is the stride
 * @param results: array where to store results, already allocated
 */
template <typename T>
using KernelFunction = void (const T*, size_t, double*);

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
    checkCudaErrors(cudaMemcpy(ptr.get(), vals, arrSize,
                               cudaMemcpyHostToDevice));
  }

  return ptr;
#else
  FLIT_UNUSED(vals);
  FLIT_UNUSED(length);
  throw std::runtime_error("Should not use makeCudaArr without CUDA enabled");
#endif
}

class TestDisabledError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

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
std::vector<double>
runKernel(KernelFunction<T>* kernel, const std::vector<T>& ti, size_t stride) {
#ifdef __CUDA__
  size_t runCount;
  if (stride < 1) { // the test takes no inputs
    runCount = 1;
  } else {
    runCount = ti.size() / stride;
  }

  std::unique_ptr<double[]> cuResults(new double[runCount]);
  // Note: __CPUKERNEL__ mode is broken by the change to run the kernel in
  // multithreaded mode.  Its compilation is broken.
  // TODO: fix __CPUKERNEL__ mode for testing.
# ifdef __CPUKERNEL__
  kernel(ti.data(), stride, cuResults);
# else  // not __CPUKERNEL__
  auto deviceVals = makeCudaArr(ti.data(), ti.size());
  auto deviceResult = makeCudaArr<double>(nullptr, runCount);
  kernel<<<runCount,1>>>(deviceVals.get(), stride, deviceResult.get());
  auto resultSize = sizeof(double) * runCount;
  checkCudaErrors(cudaMemcpy(cuResults.get(), deviceResult.get(), resultSize,
                             cudaMemcpyDeviceToHost));
# endif // __CPUKERNEL__
  std::vector<double> results(cuResults, cuResults + runCount);
  return results;
#else   // not __CUDA__
  // Do nothing
  FLIT_UNUSED(kernel);
  FLIT_UNUSED(ti);
  FLIT_UNUSED(stride);
  return {};
#endif  // __CUDA__
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
   *   from only a std::vector of inputs for each result pair.
   *
   * @see getInputsPerRun
   */
  virtual std::vector<TestResult> run(const std::vector<T>& ti,
                                      const std::string &filebase,
                                      const bool shouldTime,
                                      const int timingLoops,
                                      const int timingRepeats) {
    FLIT_UNUSED(timingRepeats);
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    std::vector<TestResult> results;
    std::vector<T> emptyInput;
    auto stride = getInputsPerRun();
    std::vector<std::vector<T>> inputSequence;

    if (stride < 1) { // the test does not take any inputs
      inputSequence.push_back(ti);
    } else {
      // Split up the input.  One for each run
      auto begin = ti.begin();
      auto runCount = ti.size() / stride;
      for (decltype(runCount) i = 0; i < runCount; i++) {
        auto end = begin + stride;
        inputSequence.emplace_back(std::vector<T>(begin, end));
        begin = end;
      }
    }

    // By default, the function to be timed is run_impl
    std::function<Variant(const std::vector<T>&)> runner;
    int runcount = 0;
    runner = [this,&runcount] (const std::vector<T>& runInput) {
      runcount++;
      return this->run_impl(runInput);
    };
#ifdef __CUDA__
    // Use the cuda kernel if it is available by replacing runner
    auto kernel = getKernel();
    if (kernel != nullptr) {
      runner = [kernel, stride, &runcount] (const std::vector<T>& ti) {
        // TODO: implement this timer better.
        runcount++;
        auto scorelist = runKernel(kernel, ti, stride);
        return Variant{ scorelist[0] };
      }
    }
#endif // __CUDA__

    // Run the tests
    struct TimedResult {
      Variant result;
      int_fast64_t time;
      std::string resultfile;

      TimedResult(Variant res, int_fast64_t t, const std::string &f = "")
        : result(res), time(t), resultfile(f) { }
    };
    std::vector<TimedResult> resultValues;

    for (size_t i = 0; i < inputSequence.size(); i++) {
      auto& runInput = inputSequence.at(i);
      Variant testResult;
      int_fast64_t timing = 0;
      if (shouldTime) {
        auto timed_runner = [runner,&runInput,&testResult] () {
          testResult = runner(runInput);
          // Throw an exception to exit early
          if (testResult.type() == Variant::Type::None) {
            throw TestDisabledError(std::string("FLiT test is disabled"));
          }
        };
        try {
          if (timingLoops < 1) {
            timing = time_function_autoloop(timed_runner, timingRepeats);
          } else {
            timing = time_function(timed_runner, timingLoops, timingRepeats);
          }
        } catch (const TestDisabledError& er) { }
      } else {
        testResult = runner(runInput);
      }
      // If the test returns a dummy value, then do not record this run.
      if (testResult.type() == Variant::Type::None) {
        continue;
      }
      std::string name = id;
      if (inputSequence.size() > 1) {
        name += "_idx" + std::to_string(i);
      }
      std::string resultfile;
      if (testResult.type() == Variant::Type::String) {
        resultfile = filebase + "_" + name + "_" + typeid(T).name() + ".dat";
        std::ofstream resultout = ofopen(resultfile);
        resultout << testResult.string();
        testResult = Variant(); // empty the result to release memory
      }
      results.emplace_back(name, typeid(T).name(), testResult,
                           timing, resultfile);
    }
    info_stream << id << "-" << typeid(T).name() << ": # runs = "
                << runcount << std::endl;
    return results;
  }

  /** Simply forwards the request to the appropriate overload of compare.
   *
   * If the types of the variants do not match, then a std::runtime_error is
   * thrown.
   */
  long double variant_compare(const Variant &ground_truth,
                              const Variant &test_results) {
    if (ground_truth.type() != test_results.type()) {
      throw std::runtime_error("Variants to compare are of different types");
    }
    long double val = 0.0;
    switch (ground_truth.type()) {
      case Variant::Type::LongDouble:
        val = this->compare(ground_truth.longDouble(),
                            test_results.longDouble());
        break;

      case Variant::Type::String:
        val = this->compare(ground_truth.string(),
                            test_results.string());
        break;

      default:
        throw std::runtime_error("Unimplemented Variant type");
    }
    return val;
  }

  /** This is a set of default inputs to use for the test
   *
   * This function should be implemented such that we can simply call this test
   * in the following way:
   *   test->run(test->getDefaultInput());
   */
  virtual std::vector<T> getDefaultInput() = 0;

  /** The number of inputs per test run
   *
   * If the test input to the public run() function has less than this value,
   * then an exception will be thrown from run().
   *
   * If the test input has many more inputs than this, then the run() function
   * will run the test as many times as it can with the amount of values given.
   * For example, if the test requires 5 inputs, and the vector of test inputs
   * passed in has 27 inputs, then the test will be run 5 times, and the last
   * two inputs will be unused.
   */
  virtual size_t getInputsPerRun() = 0;

  /** Custom comparison methods
   *
   * These comparison operations are meant to create a metric between the test
   * results from this test in the current compilation, and the results from
   * the ground truth compilation.  You can do things like the relative error
   * or the absolute error (for the case of long double).
   *
   * The below specified functions are the default implementations defined in
   * the base class.  It is safe to delete these two functions if this
   * implementation is adequate for you.
   *
   * Which one is used depends on the type of Variant that is returned from the
   * run_impl function.  The value returned by compare will be the value stored
   * in the database for later analysis.
   *
   * Note: when using the CUDA kernel functionality, only long double return
   * values are valid for now.
   */
  virtual long double compare(long double ground_truth,
                              long double test_results) const {
    // absolute error
    return test_results - ground_truth;
  }

  /** There is no good default implementation comparing two strings */
  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const {
    FLIT_UNUSED(ground_truth);
    FLIT_UNUSED(test_results);
    return 0.0;
  }

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
   * @return a single result.  You can return any type supported by
   *   flit::Variant.
   *
   * The returned value (whichever type is chosen) will be used by the public
   * virtual compare() method.
   */
  virtual Variant run_impl(const std::vector<T>& ti) = 0;

protected:
  const std::string id;
};

/** A completely empty test that outputs nothing */
template <typename T>
class NullTest : public TestBase<T> {
public:
  NullTest(std::string id) : TestBase<T>(std::move(id)) {}
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<TestResult> run(
      const std::vector<T>&, const bool, const size_t) override { return {}; }
protected:
  virtual KernelFunction<T>* getKernel() override { return nullptr; }
  virtual Variant run_impl(const std::vector<T>&) override { return {}; }
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
  class klass##Factory : public flit::TestFactory {         \
  public:                                                   \
    klass##Factory() {                                      \
      flit::registerTest(#klass, this);                     \
    }                                                       \
  protected:                                                \
    virtual createType create() override {                  \
      return std::make_tuple(                               \
          std::make_shared<klass<float>>(#klass),           \
          std::make_shared<klass<double>>(#klass),          \
          /* empty test for long double */                  \
          std::make_shared<flit::NullTest<long double>>(#klass) \
          );                                                \
    }                                                       \
  };                                                        \
  static klass##Factory global_##klass##Factory;            \

#else // not __CUDA__

#define REGISTER_TYPE(klass)                                \
  class klass##Factory : public flit::TestFactory {         \
  public:                                                   \
    klass##Factory() {                                      \
      flit::registerTest(#klass, this);                     \
    }                                                       \
  protected:                                                \
    virtual createType create() override {                  \
      return std::make_tuple(                               \
          std::make_shared<klass<float>>(#klass),           \
          std::make_shared<klass<double>>(#klass),          \
          std::make_shared<klass<long double>>(#klass)      \
          );                                                \
    }                                                       \
  };                                                        \
  static klass##Factory global_##klass##Factory;            \

#endif // __CUDA__

std::map<std::string, TestFactory*>& getTests();

inline void registerTest(const std::string& name, TestFactory *factory) {
  getTests()[name] = factory;
}

} // end namespace flit

#endif // TEST_BASE_HPP
