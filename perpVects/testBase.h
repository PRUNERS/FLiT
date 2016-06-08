// This contains the base class for tests.
// It includes a 'register' static method for a factory style
// instantiation of tests.

#ifndef TESTBASE
#define TESTBASE

#include <map>
#include <string>

#include "QFPHelpers.h"

namespace QFPTest {
  
void setWatching(bool watch = true);

typedef std::pair<std::string, std::pair<long double, long double> > resultType;

  
struct testInput {
  size_t iters;
  size_t highestDim;
  size_t ulp_inc;
  float min;
  float max;
  QFPHelpers::sort_t reduction_sort_type;
};

template<typename T>
void pushWatchData();

template<typename T>
void popWatchData();  

template<typename T>
volatile T&  getWatchData();

class TestBase;

class TestFactory{
 public:
  virtual TestBase *create() = 0;
};

class TestBase {
public:
 TestBase(std::string id):id(id){}
  static inline
    void registerTest(const std::string& name, TestFactory *factory){
    tests[name] = factory;
  }
  
  virtual resultType floatTest(const testInput&) = 0;
  virtual resultType doubleTest(const testInput&) = 0;
  virtual resultType longTest(const testInput&) = 0;
  static std::map<std::string, TestFactory*> tests;  
  const std::string id;
};

 
#define REGISTER_TYPE(klass) \
  class klass##Factory : public TestFactory  {			  \
  public:					  \
    klass##Factory() {				  \
      TestBase::registerTest(#klass, this);	  \
    } \
    virtual TestBase *create(){ \
      return new klass(); \
    } \
  }; \
  static klass##Factory global_##klass##Factory;
}

#define FUNC_OP_OVERRIDE_CALLERS(klass)	\
  klass():TestBase(#klass){} \
  resultType floatTest(const testInput& ti) override { \
    return operator()<float>(ti); \
  } \
  resultType doubleTest(const testInput& ti) override { \
    return operator()<double>(ti); \
  } \
  resultType longTest(const testInput& ti) override { \
    return operator()<long double>(ti); \
  } \

#endif
