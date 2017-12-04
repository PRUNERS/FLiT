/**
 * A simple test harness for running tests against the FLiT framework.
 *
 * This framework is to create a single test executable for each test source
 * file, very much like how it is suggested to do with QtTest.  This framework
 * is implemented as simply as it can be so that it can be verified by
 * inspection and not require heavy testing.
 *
 * Usage instructions:
 *
 * Include it as usual with
 *
 *   #include "test_harness.h"
 *
 * This file define the main() function as well as the following functions:
 *
 * TH_VERIFY(bool)                  - your standard assertion
 * TH_EQUAL(actual, expected)       - assert (actual == expected)
 * TH_NOT_EQUAL(actual, unexpected) - assert (actual != expected)
 * TH_FAIL(description)             - a failed assertion with a description
 * TH_SKIP(description)             - exit the test early with a description
 * TH_THROWS(expression, exception) - fail unless the expression throws
 *                                    exception
 *
 * These macros can be called from the top-level or from helper functions
 *
 * A test is simply a void function taking no arguments.  It is registered with
 * the macro
 *
 * TH_REGISTER(test_name)
 *
 * Here is an example:
 *
 *   #include "test_harness.h"
 *   void test_add() {
 *     TH_EQUAL(1 + 2, 3);
 *   }
 *   TH_REGISTER(test_add)
 *
 * That is all that is required to implement and add a test.  Tests will
 * execute in alphabetical order.
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

// Assertion definitions
#define TH_VERIFY_MSG(x, msg) \
  if (!(x)) { \
    throw th::AssertionError(__func__, __LINE__, msg);\
  }
#define TH_VERIFY(x) TH_VERIFY_MSG(x, "TH_VERIFY("#x")")
#define TH_EQUAL(a, b) TH_VERIFY_MSG((a) == (b), "TH_EQUAL("#a", "#b")")
#define TH_NOT_EQUAL(a, b) TH_VERIFY_MSG((a) != (b), "TH_NOT_EQUAL("#a", "#b")")
#define TH_FAIL(msg) \
  TH_VERIFY_MSG(false, std::string("TH_FAIL(\"") + msg + "\")")
#define TH_SKIP(msg) throw th::SkipError(__func__, __LINE__, msg)
#define TH_THROWS(exp, exception)                            \
  try {                                                      \
    exp;                                                     \
    TH_VERIFY_MSG(false, #exp " did not throw " #exception); \
  } catch (exception&) {}
// Adds the test to map th::tests before main() is called
#define TH_REGISTER(test_name)             \
  struct TH_Registered_Test_##test_name {  \
    TH_Registered_Test_##test_name() {     \
      th::tests[#test_name] = test_name;   \
    }                                      \
  };                                       \
  TH_Registered_Test_##test_name r_##test_name

// includes
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

// namespace definitions
namespace th {
  using test = void (*)(void);
  std::string current_test;
  std::map<std::string, th::test> tests;

  // Signals an assertion failure
  class AssertionError : public std::exception {
  public:
    AssertionError(std::string func, long line, std::string msg)
      : _func(func), _line(line), _msg(msg)
    {
      std::ostringstream msg_stream;
      msg_stream
        << "Assertion failure: " << _func << ":" << _line << " - " << _msg;
      _what = msg_stream.str();
    }
    virtual const char* what() const noexcept override {
      return _what.c_str();
    }
  protected:
    std::string _func;
    long _line;
    std::string _msg;
    std::string _what;
  };

  // Signals a test that is skipped
  class SkipError : public AssertionError {
  public:
    SkipError(std::string func, long line, std::string msg)
      : AssertionError(func, line, msg)
    {
      std::ostringstream msg_stream;
      msg_stream
        << "Test skipped: " << _func << ":" << _line << " - " << _msg;
      _what = msg_stream.str();
    }
  };

};

int main(int argCount, char *argList[]) {
  bool quiet = false;
  if (argCount > 1 &&
      (argList[1] == std::string("--quiet") ||
       argList[1] == std::string("-q")))
  {
    quiet = true;
  }
  std::vector<std::string> failed_tests;
  std::vector<std::string> skipped_tests;

  for (auto &test_pair : th::tests) {
    auto &test_name = test_pair.first;
    auto &test_ptr = test_pair.second;
    th::current_test = test_name;
    try {
      test_ptr();
    } catch (const th::SkipError &err) {
      if (!quiet) {
        std::cout << test_name << ": " << err.what() << "\n";
      }
      skipped_tests.emplace_back(test_name);
    } catch (const th::AssertionError &err) {
      std::cout << test_name << ": " << err.what() << "\n";
      failed_tests.emplace_back(test_name);
    } catch (const std::exception &err) {
      std::cout << test_name << ": Uncought exception - "
                << typeid(err).name() << ": " << err.what() << "\n";
      failed_tests.emplace_back(test_name);
    } catch (...) {
      auto err = std::current_exception;
      std::cout << test_name << ": Uncought throw (not a std::exception)\n";
      failed_tests.emplace_back(test_name);
    }
  }

  // print results
  if (!quiet) {
    if (failed_tests.size() > 0 || skipped_tests.size() > 0) {
      std::cout << "\n----------------------------------------"
                   "----------------------------------------\n\n";
    }
    if (failed_tests.size() > 0) {
      std::cout << "Failed tests:\n";
      for (auto &test_name : failed_tests) {
        std::cout << "  " << test_name << std::endl;
      }
      std::cout << "\n";
    }
    if (skipped_tests.size() > 0) {
      std::cout << "Skipped tests:\n";
      for (auto &test_name : skipped_tests) {
        std::cout << "  " << test_name << std::endl;
      }
      std::cout << "\n";
    }

    int test_successes = th::tests.size() - failed_tests.size()
                         - skipped_tests.size();
    std::cout << "Test Results:\n"
              << "  failures:   " << failed_tests.size() << std::endl
              << "  successes:  " << test_successes << std::endl
              << "  skips:      " << skipped_tests.size() << std::endl;
  }

  return failed_tests.size();
}

#endif // TEST_HARNESS_H

