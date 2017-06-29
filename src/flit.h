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

namespace flit {

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
  std::string groundTruth = "";   // input for ground-truth comparison

  /** Give a string representation of this struct for printing purposes */
  std::string toString();
private:
  /** Convert a bool to a string */
  static inline std::string boolToString(bool boolean) {
    return (boolean ? "true" : "false");
  }
};

/** Parse arguments */
FlitOptions parseArguments(int argCount, char* argList[]);

/** Returns the usage information as a string */
std::string usage(std::string progName);

/** Parse the results file into a vector of results */
std::vector<TestResult> parseResults(std::istream &in);

inline void outputResults (const std::vector<TestResult>& results,
    std::ostream& out)
{
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
         "comparison,"
         "comparison_d,"
         "file,"
         "nanosec"
      << std::endl;
  for(const auto& result: results){
    out
      << result.name() << ","                        // test case name
      << FLIT_HOST << ","                            // hostname
      << FLIT_COMPILER << ","                        // compiler
      << FLIT_OPTL << ","                            // optimization level
      << FLIT_SWITCHES << ","                        // compiler flags
      << result.precision() << ","                   // precision
      ;

    if (result.result().type() == Variant::Type::LongDouble) {
      out
        << as_int(result.result().longDouble()) << "," // score
        << result.result().longDouble() << ","       // score_d
        ;
    } else {
      out
        << FLIT_NULL << ","                          // score
        << FLIT_NULL << ","                          // score_d
        ;
    }

    if (result.resultfile().empty()) {
      out << FLIT_NULL << ",";                       // resultfile
    } else {
      out << result.resultfile() << ",";             // resultfile
    }

    if (result.is_comparison_null()) {
      out
        << FLIT_NULL << ","                          // comparison
        << FLIT_NULL << ","                          // comparison_d
        ;
    } else {
      out
        << as_int(result.comparison()) << ","        // comparison
        << result.comparison() << ","                // comparison_d
        ;
    }

    out
      << FLIT_FILENAME << ","                        // executable filename
      << result.nanosecs()                           // nanoseconds
      << std::endl;
  }
}


template <typename F>
void runTestWithDefaultInput(TestFactory* factory,
                             std::vector<TestResult>& totResults,
                             bool shouldTime = true,
                             int timingLoops = 1) {
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto results = test->run(ip, shouldTime, timingLoops);
  totResults.insert(totResults.end(), results.begin(), results.end());
  info_stream.flushout();
}

template <typename F>
long double runComparison_impl(TestFactory* factory, const TestResult &gt,
                               const TestResult &res) {
  auto test = factory->get<F>();
  if (res.result().type() == Variant::Type::String) {
    return test->compare(gt.result().string(), res.result().string());
  } else if (res.result().type() == Variant::Type::LongDouble) {
    return test->compare(gt.result().longDouble(), res.result().longDouble());
  } else { throw std::runtime_error("Unsupported variant type"); }
}

inline long double runComparison(TestFactory* factory, const TestResult &gt,
                                 const TestResult &res) {
  // TODO: after moving to lazy file load, load file contents at comparison
  if (res.precision() == "f") {
    return runComparison_impl<float>(factory, gt, res);
  } else if (res.precision() == "d") {
    return runComparison_impl<double>(factory, gt, res);
  } else if (res.precision() == "e") {
    return runComparison_impl<long double>(factory, gt, res);
  } else { throw std::runtime_error("Unrecognized precision encountered"); }
}

    
    
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
    for (auto& test : getKeys(getTests())) {
      std::cout << test << std::endl;
    }
    return 0;
  }

  if (options.verbose) {
    info_stream.show();
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
  info_stream.precision(1000);

#ifdef __CUDA__
  initDeviceData();
#endif

  std::vector<TestResult> results;
  std::vector<TestResult> groundTruthResults;
  if (!options.groundTruth.empty()) {
    std::ifstream gtfile(options.groundTruth);
    // TODO: only load file contents at time of comparison
    groundTruthResults = parseResults(gtfile);
  }

  auto testMap = getTests();
  for (auto& testName : options.tests) {
    auto factory = testMap[testName];
    if (options.precision == "all" || options.precision == "float") {
      runTestWithDefaultInput<float>(factory, results, options.timing,
                                     options.timingLoops);
      //runTestComparison<float>(factory, results, options.groundTruth);
    }
    if (options.precision == "all" || options.precision == "double") {
      runTestWithDefaultInput<double>(factory, results, options.timing,
                                      options.timingLoops);
    }
    if (options.precision == "all" || options.precision == "long double") {
      runTestWithDefaultInput<long double>(factory, results, options.timing,
                                           options.timingLoops);
    }
    // TODO: dump string result to file because we might run out of memory
  }
#if defined(__CUDA__) && !defined(__CPUKERNEL__)
  cudaDeviceSynchronize();
#endif

  // Sort the results first by name then by precision
  auto testComparator = [](const TestResult &a, const TestResult &b) {
    if (a.name() != b.name()) {
      return a.name() < b.name();
    } else {
      return a.precision() < b.precision();
    }
  };
  std::sort(results.begin(), results.end(), testComparator);
  std::sort(groundTruthResults.begin(), groundTruthResults.end(),
            testComparator);

  // Let's now run the ground-truth comparisons
  if (groundTruthResults.size() > 0) {
    for (auto& res : results) {
      auto factory = testMap[res.name()];
      // Use binary search to find the first associated ground truth element
      auto gtIter = std::lower_bound(groundTruthResults.begin(),
                                     groundTruthResults.end(), res,
                                     testComparator);
      // Compare the two results if the element was found
      if (gtIter != groundTruthResults.end() &&
          res.name() == (*gtIter).name() &&
          res.precision() == (*gtIter).precision())
      {
        res.set_comparison(runComparison(factory, *gtIter, res));
      }
    }
  }
  
  // Output string-type results to individual files
  for (auto& result : results) {
    if (result.result().type() == Variant::Type::String) {
      std::string test_result_fname =
          test_result_filebase + "_"
          + result.name() + "_"
          + result.precision()
          + ".dat";
      std::ofstream test_result_out(test_result_fname);
      test_result_out << result.result().string();
      result.set_resultfile(test_result_fname);
    }
  }

  // Create the main results output
  outputResults(results, *outstream);
  return 0;
}

} // end of namespace flit

#endif // FLIT_H
