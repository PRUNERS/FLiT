// This is the main implementation, handling cmd line params and
// running the tests!

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include <cassert>
#include <cstring>

#include "flit.h"

#include "flitHelpers.h"
#include "TestBase.h"

namespace {

/** Helper class for Csv.
 *
 * Represents a single row either indexed by number or by column name.
 */
class CsvRow : public std::vector<std::string> {
public:
  const CsvRow* header() const { return m_header; }
  void setHeader(CsvRow* head) { m_header = head; }

  using std::vector<std::string>::operator[];
  std::string const& operator[](std::string col) const {
    auto iter = std::find(m_header->begin(), m_header->end(), col);
    if (iter == m_header->end()) {
      std::stringstream message;
      message << "No column named " << col;
      throw std::invalid_argument(message.str());
    }
    auto idx = iter - m_header->begin();
    return this->operator[](idx);
  }

private:
  CsvRow* m_header {nullptr};  // not owned by this class
};

/** Class for parsing csv files */
class Csv {
public:
  Csv(std::istream &in) : m_header(Csv::parseRow(in)), m_in(in) {
    m_header.setHeader(&m_header);
  }

  Csv& operator>> (CsvRow& row) {
    row = Csv::parseRow(m_in);
    row.setHeader(&m_header);
    return *this;
  }

  operator bool() const { return static_cast<bool>(m_in); }
  
private:
  static CsvRow parseRow(std::istream &in) {
    std::string line;
    std::getline(in, line);

    std::stringstream lineStream(line);
    std::string token;

    // tokenize on ','
    CsvRow row;
    while(std::getline(lineStream, token, ',')) {
      row.emplace_back(token);
    }

    // check for trailing comma with no data after it
    if (!lineStream && token.empty()) {
      row.emplace_back("");
    }

    return row;
  }

private:
  CsvRow m_header;
  std::istream &m_in;
};

/** Returns true if the element is in the container */
template<typename Container, typename Element>
bool isIn(Container c, Element e) {
  return std::find(std::begin(c), std::end(c), e) != std::end(c);
}

} // end of unnamed namespace

namespace flit {

std::string FlitOptions::toString() const {
  std::ostringstream messanger;
  messanger
    << "Options:\n"
    << "  help:           " << boolToString(this->help) << "\n"
    << "  verbose:        " << boolToString(this->verbose) << "\n"
    << "  timing:         " << boolToString(this->timing) << "\n"
    << "  timingLoops:    " << this->timingLoops << "\n"
    << "  timingRepeats:  " << this->timingRepeats << "\n"
    << "  listTests:      " << boolToString(this->listTests) << "\n"
    << "  precision:      " << this->precision << "\n"
    << "  output:         " << this->output << "\n"
    << "  compareMode:    " << boolToString(this->compareMode) << "\n"
    << "  tests:\n";
  for (auto& test : this->tests) {
    messanger << "    " << test << "\n";
  }
  messanger
    << "  compareFiles:\n";
  for (auto& filename : this->compareFiles) {
    messanger << "    " << filename << "\n";
  }
  return messanger.str();
}

FlitOptions parseArguments(int argCount, char* argList[]) {
  FlitOptions options;

  std::vector<std::string> helpOpts          = { "-h", "--help" };
  std::vector<std::string> verboseOpts       = { "-v", "--verbose" };
  std::vector<std::string> timingOpts        = { "-t", "--timing" };
  std::vector<std::string> loopsOpts         = { "-l", "--timing-loops" };
  std::vector<std::string> repeatsOpts       = { "-r", "--timing-repeats" };
  std::vector<std::string> listTestsOpts     = { "-L", "--list-tests" };
  std::vector<std::string> precisionOpts     = { "-p", "--precision" };
  std::vector<std::string> outputOpts        = { "-o", "--output" };
  std::vector<std::string> compareMode       = { "-c", "--compare-mode" };
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
    } else if (isIn(repeatsOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      try {
        options.timingRepeats = std::stoi(argList[++i]);
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
    } else if (isIn(outputOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      options.output = argList[++i];
    } else if (isIn(compareMode, current)) {
      options.compareMode = true;
    } else {
      options.tests.push_back(current);
      if (!options.compareMode && !isIn(allowedTests, current)) {
        throw ParseException("unknown test " + current);
      }
    }
  }

  // names passed on the command line in compareMode are compareFiles not tests
  if (options.compareMode) {
    options.tests.swap(options.compareFiles);
    options.tests.emplace_back("all");
    if (options.compareFiles.size() == 0) {
      throw ParseException("You must pass in some test results in compare"
                           " mode");
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
    << "  " << progName << " [options] [<test> ...]\n"
    << "  " << progName << " --compare-mode <csvfile> [<csvfile> ...]\n"
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
       "                  default is to determine automaticall the number of\n"
       "                  loops to run, specifically tuned for each\n"
       "                  individual test.  This means each test will be run\n"
       "                  multiple times by default.\n"
       "\n"
       "  -r REPEATS, --timing-repeats REPEATS\n"
       "                  How many times to repeat the timing.  The default\n"
       "                  is to repeat timing 3 times.  The timing reported\n"
       "                  is the one that ran the fastest of all of the\n"
       "                  repeats.\n"
       "\n"
       "                  Note: since repeats defaults to 3 and loops\n"
       "                  default to be automatically determine, executing\n"
       "                  a set of tests may take much longer than the user\n"
       "                  anticipates, unless loops and repeats are set by\n"
       "                  the user to a smaller value.  Or you could turn\n"
       "                  off the timing with --no-timing.\n"
       "\n"
       "  -o OUTFILE, --output OUTFILE\n"
       "                  Output test results to the given file.  All other\n"
       "                  standard output will still go to the terminal.\n"
       "                  The default behavior is to output to stdout.\n"
       "\n"
       "  -c, --compare-mode\n"
       "                  This option only makes sense to use on the ground\n"
       "                  truth executable.  You will no longer be able to\n"
       "                  pass in particular tests to execute because the\n"
       "                  arguments are interpreted as the results files to\n"
       "                  use in the comparison.\n"
       "\n"
       "                  Note: for tests returning a string, the results\n"
       "                  file will contain a relative path to the file that\n"
       "                  actually contains the string return value.  So you\n"
       "                  will want to make sure to call this option in the\n"
       "                  same directory used when executing the test\n"
       "                  executable.\n"
       "\n"
       "  -p PRECISION, --precision PRECISION\n"
       "                  Which precision to run.  The choices are 'float',\n"
       "                  'double', 'long double', and 'all'.  The default\n"
       "                  is 'all' which runs all of them.\n"
       "\n";
  return messanger.str();
}

std::string readFile(const std::string &filename) {
  std::ifstream filein(filename);
  std::stringstream buffer;
  buffer << filein.rdbuf();
  return buffer.str();
}

std::vector<TestResult> parseResults(std::istream &in) {
  std::vector<TestResult> results;

  Csv csv(in);
  CsvRow row;
  while (csv >> row) {
    auto nanosec = std::stol(row["nanosec"]);
    Variant value;
    std::string resultfile;
    if (row["score"] != "NULL") {
      // Convert score into a long double
      value = as_float(flit::stouint128(row["score"]));
    } else {
      // Read string from the resultfile
      assert(row["resultfile"] != "NULL");
      resultfile = row["resultfile"];
    }

    results.emplace_back(row["name"], row["precision"], value, nanosec,
                         resultfile);
  }

  return results;
}

std::unordered_map<std::string, std::string> parseMetadata(std::istream &in) {
  std::unordered_map<std::string, std::string> metadata;

  const std::string metadataKeys[] = {
    "host",
    "compiler",
    "optl",
    "switches",
    "file"
  };

  Csv csv(in);
  CsvRow row;
  if (csv >> row) {
    for (auto key : metadataKeys) {
      metadata.emplace(key, row[key]);
    }
  }

  return metadata;
}

std::string removeIdxFromName(const std::string &name) {
  std::string pattern("_idx"); // followed by 1 or more digits
  auto it = std::find_end(name.begin(), name.end(),
                          pattern.begin(), pattern.end());
  // assert that after the pattern, all the remaining chars are digits.
  assert(it == name.end() ||
         std::all_of(it + pattern.size(), name.end(), [](char c) {
           return '0' <= c && c <= '9';
         }));
  return std::string(name.begin(), it);
}

} // end of namespace flit
