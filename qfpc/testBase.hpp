// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TEST_BASE_HPP
#define TEST_BASE_HPP

// TODO: Move calls to REGISTER_TYPE to a single C++ file
// TODO: Change the tests to all .hpp files included by this single C++ file

#include "QFPHelpers.hpp"

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
    auto begin = ti.vals.begin();
    for (decltype(runCount) i = 0; i < runCount; i++) {
      auto end = begin + stride;

      TestInput<T> runInput {ti.iters, ti.highestDim, ti.ulp_inc, ti.min, ti.max, {}};
      runInput.vals = std::vector<T>(begin, end);

      auto scores = run_impl(runInput);
      std::string name = id;
      if (runCount != 1) {
        name += "_idx" + std::to_string(i);
      }
      results.insert({{name, typeid(T).name()}, scores});
      begin = end;
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

inline std::map<std::string, TestFactory*>& getTests() {
  static std::map<std::string, TestFactory*> tests;
  return tests;
}

inline void registerTest(const std::string& name, TestFactory *factory) {
  getTests()[name] = factory;
}

}

#define REGISTER_TYPE(klass)                             \
  class klass##Factory : public QFPTest::TestFactory {   \
  public:                                                \
    klass##Factory() {                                   \
      QFPTest::registerTest(#klass, this);               \
    }                                                    \
  protected:                                             \
    virtual createType create() {                        \
      return std::make_tuple(                            \
          std::make_shared<klass<float>>(#klass),        \
          std::make_shared<klass<double>>(#klass),       \
          std::make_shared<klass<long double>>(#klass)   \
          );                                             \
    }                                                    \
  };                                                     \
  static klass##Factory global_##klass##Factory;         \

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
    virtual QFPTest::ResultType::mapped_type             \
    run_impl(const QFPTest::TestInput<T>&) { return {}; }\
  };                                                     \


#endif // TEST_BASE_HPP
