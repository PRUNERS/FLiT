// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TESTBASE
#define TESTBASE

#include <map>
#include <string>
#include <utility>
#include <vector>

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

class TestBase;

class TestFactory {
public:
  virtual std::vector<TestBase *> create() = 0;
};

class TestBase {
public:
  TestBase(std::string id):id(id) {}
  virtual ~TestBase() = default;
  static inline
  void registerTest(const std::string& name, TestFactory *factory) {
    getTests()[name] = factory;
  }
  virtual resultType operator()(const testInput&) = 0;
  static std::map<std::string, TestFactory*>& getTests() {
    static std::map<std::string, TestFactory*> tests;
    return tests;
  }
  const std::string id;
};

}

#define REGISTER_TYPE(klass) \
  class klass##Factory : public QFPTest::TestFactory {   \
  public:                                                \
    klass##Factory() {                                   \
      QFPTest::TestBase::registerTest(#klass, this);     \
    }                                                    \
    virtual std::vector<QFPTest::TestBase *> create() {  \
      return {                                           \
          new klass<float>(#klass),                      \
          new klass<double>(#klass),                     \
          new klass<long double>(#klass)                 \
      };                                                 \
    }                                                    \
  };                                                     \
  static klass##Factory global_##klass##Factory;         \

#define EIGEN_CLASS_DEF(klass, file)                     \
  template <typename T>                                  \
  class klass : public QFPTest::TestBase{                \
  public:                                                \
    klass(std::string id) : QFPTest::TestBase(id){}      \
    QFPTest::resultType                                  \
    operator()(const QFPTest::testInput& ti) {           \
      Q_UNUSED(ti);                                      \
      if(sizeof(T) != 4) return {};                      \
      auto fileS = std::string(#file);                   \
      g_test_stack_mutex.lock();                         \
      g_test_stack[fileS];                               \
      g_test_stack_mutex.unlock();                       \
      eigenResults_mutex.lock();                         \
      eigenResults[fileS];                               \
      eigenResults_mutex.unlock();                       \
      test_##file();                                     \
      g_test_stack[fileS].clear();                       \
      auto res = eigenResults[fileS];                    \
      eigenResults[fileS].clear();                       \
      return res;                                        \
    }                                                    \
  };                                                     \


#endif
