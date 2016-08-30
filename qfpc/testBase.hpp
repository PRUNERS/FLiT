// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TEST_BASE_HPP
#define TEST_BASE_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include <memory>

#include "QFPHelpers.hpp"

namespace QFPTest {

void setWatching(bool watch = true);


using resultType = std::map<std::pair<const std::string, const std::string>,
                            std::pair<long double, long double>>;


std::ostream&
operator<<(std::ostream&, const resultType&);

struct testInput {
  size_t iters;
  size_t highestDim;
  size_t ulp_inc;
  float min;
  float max;
};

template<typename T>
void pushWatchData();

template<typename T>
void popWatchData();

template<typename T>
volatile T&  getWatchData();

template <typename T>
class TestBase {
public:
  TestBase(std::string id) : id(std::move(id)) {}
  virtual ~TestBase() = default;
  virtual resultType operator()(const testInput&) = 0;
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

#define EIGEN_CLASS_DEF(klass, file)                     \
  template <typename T>                                  \
  class klass : public QFPTest::TestBase<T> {            \
  public:                                                \
    klass(std::string id)                                \
      : QFPTest::TestBase<T>(std::move(id)) {}           \
    QFPTest::resultType                                  \
    operator()(const QFPTest::testInput& ti) {           \
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
  };                                                     \


#endif // TEST_BASE_HPP
