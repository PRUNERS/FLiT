/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

// This is the main implementation, handling cmd line params and
// running the tests!

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <sstream>
#include <type_traits>
#include <typeinfo>

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
  // Inherit base class constructors
  using std::vector<std::string>::vector;

  const CsvRow* header() const { return m_header; }
  void setHeader(CsvRow* head) { m_header = head; }

  using std::vector<std::string>::operator[];
  std::string const& operator[](std::string col) const {
    if (m_header == nullptr) {
      throw std::logic_error("No header defined");
    }
    auto iter = std::find(m_header->begin(), m_header->end(), col);
    if (iter == m_header->end()) {
      std::stringstream message;
      message << "No column named " << col;
      throw std::invalid_argument(message.str());
    }
    auto idx = iter - m_header->begin();
    return this->at(idx);
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
    << "  info:           " << boolToString(this->info) << "\n"
    << "  verbose:        " << boolToString(this->verbose) << "\n"
    << "  timing:         " << boolToString(this->timing) << "\n"
    << "  timingLoops:    " << this->timingLoops << "\n"
    << "  timingRepeats:  " << this->timingRepeats << "\n"
    << "  listTests:      " << boolToString(this->listTests) << "\n"
    << "  precision:      " << this->precision << "\n"
    << "  output:         " << this->output << "\n"
    << "  compareMode:    " << boolToString(this->compareMode) << "\n"
    << "  compareGtFile:  " << this->compareGtFile << "\n"
    << "  compareSuffix:  " << this->compareSuffix << "\n"
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

FlitOptions parseArguments(int argCount, char const* const* argList) {
  FlitOptions options;

  std::vector<std::string> helpOpts          = { "-h", "--help" };
  std::vector<std::string> infoOpts          = { "--info" };
  std::vector<std::string> verboseOpts       = { "-v", "--verbose" };
  std::vector<std::string> timingOpts        = { "-t", "--timing" };
  std::vector<std::string> loopsOpts         = { "-l", "--timing-loops" };
  std::vector<std::string> repeatsOpts       = { "-r", "--timing-repeats" };
  std::vector<std::string> listTestsOpts     = { "-L", "--list-tests" };
  std::vector<std::string> precisionOpts     = { "-p", "--precision" };
  std::vector<std::string> outputOpts        = { "-o", "--output" };
  std::vector<std::string> compareMode       = { "-c", "--compare-mode" };
  std::vector<std::string> compareGtFileOpts = { "-g", "--compare-gt" };
  std::vector<std::string> compareSuffixOpts = { "-s", "--suffix" };
  std::vector<std::string> allowedPrecisions = {
    "all", "float", "double", "long double"
  };
  auto allowedTests = getKeys(getTests());
  allowedTests.emplace_back("all");
  for (int i = 1; i < argCount; i++) {
    std::string current(argList[i]);
    if (isIn(helpOpts, current)) {
      options.help = true;
    } else if (isIn(infoOpts, current)) {
      options.info = true;
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
      options.timing = false;
    } else if (isIn(compareGtFileOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      options.compareGtFile = argList[++i];
    } else if (isIn(compareSuffixOpts, current)) {
      if (i+1 == argCount) {
        throw ParseException(current + " requires an argument");
      }
      options.compareSuffix = argList[++i];
    } else {
      options.tests.push_back(current);
      if (!options.compareMode && !isIn(allowedTests, removeIdxFromName(current))) {
        throw ParseException("unknown test " + current);
      }
    }
  }

  // names passed on the command line in compareMode are compareFiles not tests
  if (options.compareMode) {
    options.tests.swap(options.compareFiles);
    if (options.compareFiles.size() == 0) {
      throw ParseException("You must pass in some test results in compare"
                           " mode");
    }
  } else if (options.tests.size() == 0
          || isIn(options.tests, std::string("all"))) {
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
       "Positional Arguments:\n"
       "\n"
       "  <test>          The name of the test as shown by the --list-tests\n"
       "                  option.  If this is not specified, then all tests\n"
       "                  will be executed.\n"
       "\n"
       "                  When a test is data-driven, it generates multiple\n"
       "                  test results.  Each of these test results will be\n"
       "                  appended with \"_idx\" followed by a number\n"
       "                  indicating which data-driven input it used.  You\n"
       "                  may specify this same suffix when you specify the\n"
       "                  test to only run particular data-driven inputs\n"
       "                  instead of all of them.\n"
       "\n"
       "                  Example:\n"
       "                    " << progName << " TestCase_idx3 TestCase_idx5\n"
       "\n"
       "                  This will only run inputs 3 and 5 instead of all\n"
       "                  of them.  Note that if you specify an index higher\n"
       "                  than the number of inputs for your test, then it\n"
       "                  will be ignored.\n"
       "\n"
       "                  Note: this is zero-based indexing.  So to run the\n"
       "                  2nd test input sequence, use TestCase_idx1.\n"
       "\n"
       "  <csvfile>       File path to the csv results to compare this\n"
       "                  executable's results against.\n"
       "\n"
       "Options:\n"
       "\n"
       "  -h, --help      Show this help and exit\n"
       "\n"
       "  --info          Show compilation information and exit\n"
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
       "                  default is to determine automatically the number\n"
       "                  of loops to run, specifically tuned for each\n"
       "                  individual test.  This means each test will be run\n"
       "                  multiple times by default.\n"
       "\n"
       "                  Note: when MPI is turned on (by setting the\n"
       "                  enable_mpi variable in flit-config.toml to true)\n"
       "                  and running more than one process, automatic loop\n"
       "                  tuning is disabled, otherwise a deadlock could\n"
       "                  occur.  If you try to use the auto-tuning (which\n"
       "                  is the default behavior), then the loops will be\n"
       "                  converted to one instead.\n"
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
       "                  This option implies --no-timing\n"
       "\n"
       "                  Note: for tests returning a string, the results\n"
       "                  file will contain a relative path to the file that\n"
       "                  actually contains the string return value.  So you\n"
       "                  will want to make sure to call this option in the\n"
       "                  same directory used when executing the test\n"
       "                  executable.\n"
       "\n"
       "  -g GT_RESULTS, --compare-gt GT_RESULTS\n"
       "                  Only applicable with --compare-mode on.\n"
       "\n"
       "                  Specify the csv file to use for the ground-truth\n"
       "                  results.  If this is not specified, then the\n"
       "                  associated tests will be executed in order to\n"
       "                  compare.  If there are tests not in this results\n"
       "                  file, but needing comparison, then those tests\n"
       "                  will be executed to be able to compare.\n"
       "\n"
       "                  If the --output option is specified as well, then\n"
       "                  only the extra tests that were executed and not\n"
       "                  found in this results file will be output.\n"
       "\n"
       "  -s SUFFIX, --suffix SUFFIX\n"
       "                  Only applicable with --compare-mode on.\n"
       "\n"
       "                  Typically in compare mode, each compare file is\n"
       "                  read in, compared, and then rewritten over the top\n"
       "                  of that compare file.  If you want to keep the\n"
       "                  compare file untouched and instead output to a\n"
       "                  different file, you can use this option to add a\n"
       "                  suffix to the compared filename.\n"
       "\n"
       "                  In other words, the default value for this option\n"
       "                  is the empty string, causing the filenames to\n"
       "                  match.\n"
       "\n"
       "  -p PRECISION, --precision PRECISION\n"
       "                  Which precision to run.  The choices are 'float',\n"
       "                  'double', 'long double', and 'all'.  The default\n"
       "                  is 'all' which runs all of them.\n"
       "\n";
  return messanger.str();
}

std::string readFile(const std::string &filename) {
  std::ifstream filein;
  filein.exceptions(std::ios::failbit);
  filein.open(filename);
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
    if (row["score_hex"] != "NULL") {
      // Convert score_hex into a long double
      value = as_float(flit::stouint128(row["score_hex"]));
    } else {
      // Read string from the resultfile
      if (row["resultfile"] == "NULL") {
        throw std::invalid_argument("must give score or resultfile");
      }
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

std::string removeIdxFromName(const std::string &name, int *idx) {
  std::string pattern("_idx"); // followed by 1 or more digits
  auto it = std::find_end(name.begin(), name.end(),
                          pattern.begin(), pattern.end());
  // make sure after the pattern, all the remaining chars are digits.
  bool is_integer_idx = \
        it == name.end() ||
        std::all_of(it + pattern.size(), name.end(), [](char c) {
           return '0' <= c && c <= '9';
        });
  if (!is_integer_idx) {
    throw std::invalid_argument("in removeIdxFromName, non-integer idx");
  }
  if (idx != nullptr) {
    if (it != name.end()) {
      *idx = std::stoi(std::string(it+4, name.end()));
    } else {
      *idx = -1;
    }
  }
  return std::string(name.begin(), it);
}

std::vector<std::string> calculateMissingComparisons(const FlitOptions &opt) {
  // We don't want to run the tests that are specified in the gt test file
  std::set<std::string> gtTestNames;
  if (!opt.compareGtFile.empty()) {
    std::ifstream gtresultfile(opt.compareGtFile);
    for (auto &result : parseResults(gtresultfile)) {
      gtTestNames.emplace(result.name());
    }
  }

  // TODO: double check that we have {testname, precision} pairings in there
  // parse the incoming files to determine which tests need to be run
  std::set<std::string> testNames;
  for (auto fname : opt.compareFiles) {
    std::ifstream resultfile(fname);
    for (auto &result : parseResults(resultfile)) {
      if (gtTestNames.end() == gtTestNames.find(result.name())) {
        testNames.emplace(result.name());
      }
    }
  }

  std::vector<std::string> missingTests(testNames.begin(), testNames.end());
  return missingTests;
}

} // end of namespace flit
