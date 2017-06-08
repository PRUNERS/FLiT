// This is the main implementation, handling cmd line params and
// running the tests!

#ifndef FLIT_H
#define FLIT_H 0

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include "QFPHelpers.hpp"
#include "TestBase.hpp"

#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.hpp"
#endif

void outputResults(const QFPTest::ResultType& scores);

template <typename F>
void runTestWithDefaultInput(QFPTest::TestFactory* factory,
                             QFPTest::ResultType& totScores,
                             bool shouldTime = true,
                             int timingLoops = 1) {
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto scores = test->run(ip, shouldTime, timingLoops);
  totScores.insert(scores.begin(), scores.end());
  QFPHelpers::info_stream.flushout();
}

/** Command-line options */
struct FlitOptions {
  bool help = false;      // show usage and exit
  bool listTests = false; // list available tests and exit
  bool verbose = false;   // show debug verbose messages
  std::vector<std::string> tests; // which tests to run
  std::string precision = "all";  // which precision to use
  bool timing = true;     // should we run timing?
  int timingLoops = 1;    // < 1 means to auto-determine the timing loops

  /** Give a string representation of this struct for printing purposes */
  std::string toString();
private:
  /** Convert a bool to a string */
  static inline std::string boolToString(bool boolean) {
    return (boolean ? "true" : "false");
  }
};

/** Returns true if the element is in the container */
template<typename Container, typename Element>
bool isIn(Container c, Element e) {
  return std::find(std::begin(c), std::end(c), e) != std::end(c);
}

/** Returns the keys of a std::map as a std::vector */
template<typename A, typename B>
std::vector<A> getKeys(std::map<A, B> map) {
  std::vector<A> keys;
  for (auto pair : map) {
    keys.emplace_back(pair.first);
  }
  return keys;
}

class ParseException : std::exception {
public:
  ParseException(const std::string& message) : _message(message) {}
  virtual const char* what() const noexcept { return _message.c_str(); }
private:
  const std::string _message;
};

/** Parse arguments */
FlitOptions parseArguments(int argCount, char* argList[]);

/** Returns the usage information as a string */
std::string usage(std::string progName);

inline int runFlitTests(int argc, char* argv[]) {
  // Argument parsing
  FlitOptions options;
  try {
    options = parseArguments(argc, argv);
  } catch (ParseException &ex) {
    std::cerr << "Error: " << ex.what() << "\n"
              << "  Use the --help option for more information\n";
    return 1;
  }

  if (options.help) {
    std::cout << usage(argv[0]);
    return 0;
  }

  if (options.listTests) {
    for (auto& test : getKeys(QFPTest::getTests())) {
      std::cout << test << std::endl;
    }
    return 0;
  }

  if (options.verbose) {
    QFPHelpers::info_stream.show();
  }

  std::cout.precision(1000); //set cout to print many decimal places
  QFPHelpers::info_stream.precision(1000);

#ifdef __CUDA__
  CUHelpers::initDeviceData();
#endif

  QFPTest::ResultType scores;
  auto testMap = QFPTest::getTests();
  for (auto& testName : options.tests) {
    auto factory = testMap[testName];
    if (options.precision == "all" || options.precision == "float") {
      runTestWithDefaultInput<float>(factory, scores, options.timing,
                                     options.timingLoops);
    }
    if (options.precision == "all" || options.precision == "double") {
      runTestWithDefaultInput<double>(factory, scores, options.timing,
                                      options.timingLoops);
    }
    if (options.precision == "all" || options.precision == "long double") {
      runTestWithDefaultInput<long double>(factory, scores, options.timing,
                                           options.timingLoops);
    }
  }
#if defined(__CUDA__) && !defined(__CPUKERNEL__)
  cudaDeviceSynchronize();
#endif

  outputResults(scores);
  return 0;
}

#endif // FLIT_H
