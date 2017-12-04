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
#include <unordered_map>
#include <utility>

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
  bool help = false;        // show usage and exit
  bool listTests = false;   // list available tests and exit
  bool verbose = false;     // show debug verbose messages
  std::vector<std::string> tests; // which tests to run
  std::string precision = "all";  // which precision to use
  std::string output = "";        // output file for results.  default stdout
  bool timing = true;       // should we run timing?
  int timingLoops = -1;     // < 1 means to auto-determine the timing loops
  int timingRepeats = 3;    // return best of this many timings
  
  bool compareMode = false; // compare results after running the test
  std::vector<std::string> compareFiles; // files for compareMode

  /** Give a string representation of this struct for printing purposes */
  std::string toString() const;
private:
  /** Convert a bool to a string */
  static inline std::string boolToString(bool boolean) {
    return (boolean ? "true" : "false");
  }
};

inline std::ostream& operator<<(std::ostream& out, const FlitOptions &opt) {
  return out << opt.toString();
}

template<typename A, typename B>
struct pair_hash {
  // This is from python's implementation of hashing a tuple
  size_t operator()(const std::pair<A, B> &thepair) const {
    std::hash<A> hasherA;
    std::hash<B> hasherB;
    size_t value = 0x345678;
    value = (1000003 * value) ^ hasherA(thepair.first);
    value = (1000003 * value) ^ hasherB(thepair.second);
    return value;
  }
};

/** Parse arguments */
FlitOptions parseArguments(int argCount, char const* const argList[]);

/** Returns the usage information as a string */
std::string usage(std::string progName);

/** Read file contents entirely into a string */
std::string readFile(const std::string &filename);

/** Parse the results file into a vector of results */
std::vector<TestResult> parseResults(std::istream &in);

/** Parse the result file to get metadata from the first row */
std::unordered_map<std::string, std::string> parseMetadata(std::istream &in);

/** Test names sometimes are postfixed with "_idx" + <num>.  Remove that postfix */
std::string removeIdxFromName(const std::string &name);

class TestResultMap {
public:
  using key_type = std::pair<std::string, std::string>;

  void loadfile(const std::string &filename) {
    std::ifstream resultfile(filename);
    auto parsed = parseResults(resultfile);
    this->extend(parsed, filename);
  }

  std::vector<TestResult*> operator[](
      const key_type &key) const
  {
    std::vector<TestResult*> all_vals;
    auto range = m_testmap.equal_range(key);
    for (auto iter = range.first; iter != range.second; iter++) {
      all_vals.push_back(iter->second);
    }
    return all_vals;
  }

  std::vector<TestResult*> fileresults(const std::string &filename) {
    std::vector<TestResult*> all_vals;
    auto range = m_filemap.equal_range(filename);
    for (auto iter = range.first; iter != range.second; iter++) {
      all_vals.push_back(&(iter->second));
    }
    return all_vals;
  }

private:
  void append(const TestResult &result, const std::string &filename) {
    auto it = m_filemap.emplace(filename, result);
    m_testmap.emplace(key_type{result.name(), result.precision()},
                      &(it->second));
  }

  void extend(const std::vector<TestResult> &results,
              const std::string &filename)
  {
    for (auto& result : results) {
      this->append(result, filename);
    }
  }

private:
  std::unordered_multimap<
    std::pair<std::string, std::string>,
    TestResult*,
    pair_hash<std::string, std::string>
    > m_testmap;   // (testname, precision) -> TestResult*
  std::unordered_multimap<std::string, TestResult> m_filemap; // filename -> TestResult
};

inline void outputResults (
    const std::vector<TestResult>& results,
    std::ostream& out,
    std::string hostname = FLIT_HOST,
    std::string compiler = FLIT_COMPILER,
    std::string optimization_level = FLIT_OPTL,
    std::string switches = FLIT_SWITCHES,
    std::string executableFilename = FLIT_FILENAME)
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
      << hostname << ","                             // hostname
      << compiler << ","                             // compiler
      << optimization_level << ","                   // optimization level
      << switches << ","                             // compiler flags
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
      << executableFilename << ","                   // executable filename
      << result.nanosecs()                           // nanoseconds
      << std::endl;
  }
}


template <typename F>
void runTestWithDefaultInput(TestFactory* factory,
                             std::vector<TestResult>& totResults,
                             const std::string &filebase = "",
                             bool shouldTime = true,
                             int timingLoops = -1,
                             int timingRepeats = 3) {
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto results = test->run(ip, filebase, shouldTime, timingLoops,
                           timingRepeats);
  totResults.insert(totResults.end(), results.begin(), results.end());
  info_stream.flushout();
}

template <typename F>
long double runComparison_impl(TestFactory* factory, const TestResult &gt,
                               const TestResult &res) {
  auto test = factory->get<F>();
  if (res.result().type() != gt.result().type()) {
    throw std::invalid_argument("Result and baseline comparison types do not"
                                " match");
  }
  if (!gt.resultfile().empty()) {
    if (gt.result().type() != Variant::Type::None) {
      throw std::invalid_argument("baseline comparison type is not None when"
                                  " the resultfile is defined");
    }
    return test->compare(readFile(gt.resultfile()),
                         readFile(res.resultfile()));
  } else if (gt.result().type() == Variant::Type::LongDouble) {
    return test->compare(gt.result().longDouble(), res.result().longDouble());
  } else { throw std::runtime_error("Unsupported variant type"); }
}

inline long double runComparison(TestFactory* factory, const TestResult &gt,
                                 const TestResult &res) {
  if (res.precision() == "f") {
    return runComparison_impl<float>(factory, gt, res);
  } else if (res.precision() == "d") {
    return runComparison_impl<double>(factory, gt, res);
  } else if (res.precision() == "e") {
    return runComparison_impl<long double>(factory, gt, res);
  } else { throw std::runtime_error("Unrecognized precision encountered"); }
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

class ParseException : public std::exception {
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
  auto testMap = getTests();
  for (auto& testName : options.tests) {
    auto factory = testMap[testName];
    if (options.precision == "all" || options.precision == "float") {
      runTestWithDefaultInput<float>(factory, results, test_result_filebase,
                                     options.timing, options.timingLoops,
                                     options.timingRepeats);
    }
    if (options.precision == "all" || options.precision == "double") {
      runTestWithDefaultInput<double>(factory, results, test_result_filebase,
                                      options.timing, options.timingLoops,
                                      options.timingRepeats);
    }
    if (options.precision == "all" || options.precision == "long double") {
      runTestWithDefaultInput<long double>(
          factory, results, test_result_filebase, options.timing,
          options.timingLoops, options.timingRepeats);
    }
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

  // Let's now run the ground-truth comparisons
  if (options.compareMode) {
    TestResultMap comparisonResults;
  
    for (auto fname : options.compareFiles) {
      comparisonResults.loadfile(fname);
    }

    // compare mode is only done in the ground truth compilation
    // so "results" are the ground truth results.
    for (auto& gtres : results) {
      auto factory = testMap[removeIdxFromName(gtres.name())];
      auto toCompare = comparisonResults[{gtres.name(), gtres.precision()}];
      for (TestResult* compResult : toCompare) {
        auto compVal = runComparison(factory, gtres, *compResult);
        compResult->set_comparison(compVal);
      }
    }

    // save back to the compare files with compare value set
    for (auto fname : options.compareFiles) {
      // read in the metadata to use in creating the file again
      std::unordered_map<std::string, std::string> metadata;
      {
        std::ifstream fin(fname);
        metadata = parseMetadata(fin);
      }

      // get all results from this file
      auto fileresultPtrs = comparisonResults.fileresults(fname);
      std::vector<TestResult> fileresults;
      for (auto resultPtr : fileresultPtrs) {
        fileresults.push_back(*resultPtr);
      }

      // sort the file results
      std::sort(fileresults.begin(), fileresults.end(), testComparator);

      // output back to a file
      {
        std::ofstream fout(fname);
        outputResults(
            fileresults,
            fout,
            metadata["host"],
            metadata["compiler"],
            metadata["optl"],
            metadata["switches"],
            metadata["file"]
            );
      }
    }
  }

  // Create the main results output
  outputResults(results, *outstream);
  return 0;
}

} // end of namespace flit

#endif // FLIT_H
