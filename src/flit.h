// This is the main implementation, handling cmd line params and
// running the tests!

#ifndef FLIT_H
#define FLIT_H 0

#include "flitHelpers.h"
#include "TestBase.h"

#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.h"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include <cstring>

// Define macros to use in the output
// These can be overridden at compile time to insert compile-time information

#ifndef FLIT_HOST
#define FLIT_HOST "HOST"
#endif // FLIT_HOST

#ifndef FLIT_COMPILER
#define FLIT_COMPILER "COMPILER"
#endif // FLIT_COMPILER

#ifndef FLIT_OPTL
#define FLIT_OPTL "OPTL"
#endif // FLIT_OPTL

#ifndef FLIT_SWITCHES
#define FLIT_SWITCHES "SWITCHES"
#endif // FLIT_SWITCHES

#ifndef FLIT_NULL
#define FLIT_NULL "NULL"
#endif // FLIT_NULL

#ifndef FLIT_FILENAME
#define FLIT_FILENAME "FILENAME"
#endif // FLIT_FILENAME

inline void outputResults (const flit::ResultType& scores, std::ostream& out) {
  using flit::operator<<;
  using flit::as_int;
  // Output the column headers
  out << "name,"
         "host,"
         "compiler,"
         "optl,"
         "switches,"
         "precision,"
         "score,"
         "score_d,"
         "resultfile,"
         "file,"
         "nanosec"
      << std::endl;
  for(const auto& i: scores){
    out
      << i.first.first << ","                        // test case name
      << FLIT_HOST << ","                            // hostname
      << FLIT_COMPILER << ","                        // compiler
      << FLIT_OPTL << ","                            // optimization level
      << FLIT_SWITCHES << ","                        // compiler flags
      << i.first.second << ","                       // precision
      ;
    auto test_result = i.second.first;
    if (test_result.type() == flit::Variant::Type::LongDouble) {
      out
        << as_int(test_result.longDouble()) << ","   // score
        << test_result.longDouble() << ","           // score_d
        << FLIT_NULL << ","                          // resultfile
        ;
    } else {
      out
        << FLIT_NULL << ","                          // score
        << FLIT_NULL << ","                          // score_d
        << test_result.string() << ","               // resultfile
        ;
    }
    out
      << FLIT_FILENAME << ","                        // executable filename
      << i.second.second                             // nanoseconds
      << std::endl;
  }
}


template <typename F>
void runTestWithDefaultInput(flit::TestFactory* factory,
                             flit::ResultType& totScores,
                             bool shouldTime = true,
                             int timingLoops = 1) {
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto scores = test->run(ip, shouldTime, timingLoops);
  totScores.insert(scores.begin(), scores.end());
  flit::info_stream.flushout();
}

/** Command-line options */
struct FlitOptions {
  bool help = false;      // show usage and exit
  bool listTests = false; // list available tests and exit
  bool verbose = false;   // show debug verbose messages
  std::vector<std::string> tests; // which tests to run
  std::string precision = "all";  // which precision to use
  std::string output = "";        // output file for results.  default stdout
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
    for (auto& test : getKeys(flit::getTests())) {
      std::cout << test << std::endl;
    }
    return 0;
  }

  if (options.verbose) {
    flit::info_stream.show();
  }

  std::unique_ptr<std::ostream> stream_deleter;
  std::ostream *outstream = &std::cout;
  std::string test_result_filebase(FLIT_FILENAME);
  if (!options.output.empty()) {
    stream_deleter.reset(new std::ofstream(options.output.c_str()));
    outstream = stream_deleter.get();
    test_result_filebase = options.output;
  }

  std::cout.precision(1000); //set cout to print many decimal places
  flit::info_stream.precision(1000);

#ifdef __CUDA__
  flit::initDeviceData();
#endif

  flit::ResultType scores;
  auto testMap = flit::getTests();
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

  // Output string-type results to individual files
  for (auto& i : scores) {
    auto test_result = i.second.first;
    auto test_name = i.first.first;
    auto precision = i.first.second;
    if (i.second.first.type() == flit::Variant::Type::String) {
      std::string test_result_fname =
          test_result_filebase + "_"
          + test_name + "_"
          + precision
          + ".dat";
      std::ofstream test_result_out(test_result_fname);
      test_result_out << test_result.string();
      i.second.first = test_result_fname;
    }
  }

  // Create the main results output
  outputResults(scores, *outstream);
  return 0;
}

#endif // FLIT_H
