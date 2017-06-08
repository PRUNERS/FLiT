// This is the main implementation, handling cmd line params and
// running the tests!

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
using namespace CUHelpers;
#endif


using namespace QFPHelpers;
using namespace QFPTest;

//this sets the test parameters
namespace {

void
outputResults(const QFPTest::ResultType& scores){
  for(const auto& i: scores){
    std::cout
      << "HOST,SWITCHES,OPTL,COMPILER,"
      << i.first.second << ",us," //sort
      << i.second.first.first << "," //score0d
      << as_int(i.second.first.first) << "," //score0
      << i.second.first.second << "," //score1d
      << as_int(i.second.first.second) << "," //score1
      << i.first.first << "," //name
      << i.second.second << "," //nanoseconds
      << "FILENAME" //filename
      << std::endl;
  }
}

template <typename F>
void runTestWithDefaultInput(QFPTest::TestFactory* factory,
                             QFPTest::ResultType& totScores,
                             bool shouldTime = true,
                             int timingLoops = 1) {
  using namespace std::chrono;
    
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto scores = test->run(ip, shouldTime, timingLoops);
  totScores.insert(scores.begin(), scores.end());
  info_stream.flushout();
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
  std::string toString() {
    std::ostringstream messanger;
    messanger
      << "Options:\n"
      << "  help:         " << boolToString(this->help) << "\n"
      << "  listTests:    " << boolToString(this->listTests) << "\n"
      << "  verbose:      " << boolToString(this->verbose) << "\n"
      << "  timing:       " << boolToString(this->timing) << "\n"
      << "  timingLoops:  " << this->timingLoops << "\n"
      << "  precision:    " << this->precision << "\n"
      << "  tests:\n";
    for (auto& test : this->tests) {
      messanger << "    " << test << "\n";
    }
    return messanger.str();
  }

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
FlitOptions parseArguments(int argCount, char* argList[]) {
  FlitOptions options;

  std::vector<std::string> helpOpts = { "-h", "--help" };
  std::vector<std::string> verboseOpts = { "-v", "--verbose" };
  std::vector<std::string> timingOpts = { "-t", "--timing" };
  std::vector<std::string> loopsOpts = { "-l", "--timing-loops" };
  std::vector<std::string> listTestsOpts = { "-L", "--list-tests" };
  std::vector<std::string> precisionOpts = { "-p", "--precision" };
  std::vector<std::string> allowedPrecisions = {
    "all", "float", "double", "long double"
  };
  auto allowedTests = getKeys(getTests());
  allowedTests.emplace_back("all");
  for (int i = 1; i < argCount; i++) {
    std::string current(argList[i]);
    if (isIn(helpOpts, current)) {
      options.help = true;
    } else if (isIn(verboseOpts, current)) {
      options.verbose = true;
    } else if (isIn(timingOpts, current)) {
      options.timing = true;
    } else if (current == "--no-timing") {
      options.timing = false;
    } else if (isIn(loopsOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      try {
        options.timingLoops = std::stoi(argList[++i]);
      } catch (std::invalid_argument &) {
        throw ParseException(std::string(argList[i]) + " is not an integer\n");
      }
    } else if (isIn(listTestsOpts, current)) {
      options.listTests = true;
    } else if (isIn(precisionOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      options.precision = argList[++i];
      if (!isIn(allowedPrecisions, options.precision)) {
        throw ParseException("unsupported precision " + options.precision);
      }
    } else {
      options.tests.push_back(current);
      if (!isIn(allowedTests, current)) {
        throw ParseException("unknown test " + current);
      }
    }
  }

  if (options.tests.size() == 0 || isIn(options.tests, std::string("all"))) {
    options.tests = getKeys(getTests());
  }

  return options;
}

/** Returns the usage information as a string */
std::string usage(std::string progName) {
  std::ostringstream messanger;
  messanger
    << "Usage:\n"
    << "  " << progName << " [options] [[test] ...]\n"
    << "\n"
       "Description:\n"
       "  Runs the FLiT tests and outputs the results to the console in CSV\n"
       "  format.\n"
       "\n"
       "Options:\n"
       "\n"
       "  -h, --help      Show this help and exit\n"
       "\n"
       "  -L, --list-tests\n"
       "                  List all available tests and exit.\n"
       "\n"
       "  -v, --verbose   Turn on the debug output from the tests (i.e. the\n"
       "                  output sent to info_stream).\n"
       "\n"
       "  -t, --timing    Turn on timing.  This is the default.\n"
       "\n"
       "  --no-timing     Turn off timing.\n"
       "\n"
       "  -l LOOPS, --timing-loops LOOPS\n"
       "                  The number of loops to run with the timing.  The\n"
       "                  default is to simply run one loop so that the timing\n"
       "                  does not slow down execution by a significant\n"
       "                  amount.  If you set it to 0 or a negative number,\n"
       "                  then the number of loops will be determined\n"
       "                  automatically and tuned for each individual test.\n"
       "\n"
       "  -p PRECISION, --precision PRECISION\n"
       "                  Which precision to run.  The choices are 'float',\n"
       "                  'double', 'long double', and 'all'.  The default\n"
       "                  is 'all' which runs all of them.\n"
       "\n";
  return messanger.str();
}

} //namespace

int runFlitTests(int argc, char* argv[]) {
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
    for (auto& test : getKeys(getTests())) {
      std::cout << test << std::endl;
    }
    return 0;
  }

  if (options.verbose) {
    info_stream.show();
  }

  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

#ifdef __CUDA__
  CUHelpers::initDeviceData();
#endif

  QFPTest::ResultType scores;
  auto testMap = getTests();
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

int main(int argCount, char* argList[]) {
  return runFlitTests(argCount, argList);
}

