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

#include "test_harness.h"

#include "flit.h"
#include "flit.cpp"

#include "TestBase.h"   // for operator<<(flit::TestResult ...)

#include <algorithm>
#include <array>
#include <ios>
#include <memory>
#include <sstream>
#include <vector>

#include <cstdio>

namespace {
struct TempFile {
public:
  std::string name;
  std::ofstream out;
  TempFile() {
    char fname_buf[L_tmpnam];
    char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
    if (s != fname_buf) {
      throw std::runtime_error("Could not create temporary file");
    }

    name = fname_buf;
    name += "-tst_flit.in";    // this makes the danger much less likely
    out.exceptions(std::ios::failbit);
    out.open(name);
  }
  ~TempFile() {
    out.close();
    std::remove(name.c_str());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T> &v) {
  out << "[";
  if (v.size() > 0) {
    out << v[0];
  }
  for (int i = 1; i < v.size(); i++) {
    out << ", " << v[i];
  }
  out << "]";
  return out;
}
} // end of unnamed namespace

namespace tst_CsvRow {
void tst_CsvRow_header() {
  CsvRow row {"1", "2", "3", "4"};
  TH_EQUAL(row.header(), nullptr);
  row.setHeader(&row);
  TH_EQUAL(row.header(), &row);
}
TH_REGISTER(tst_CsvRow_header);

void tst_CsvRow_operator_brackets() {
  CsvRow row {"1", "2", "3", "4"};
  CsvRow header {"a", "b", "c", "d"};
  row.setHeader(&header);
  TH_EQUAL(row["a"], "1");
  TH_EQUAL(row["b"], "2");
  TH_EQUAL(row["c"], "3");
  TH_EQUAL(row["d"], "4");
  TH_THROWS(row["Mike"], std::invalid_argument);

  // Row missing elements
  header.emplace_back("e");
  TH_THROWS(row["e"], std::out_of_range);

  // null header
  row.setHeader(nullptr);
  TH_THROWS(row["a"], std::logic_error);
}
TH_REGISTER(tst_CsvRow_operator_brackets);
} // end of namespace tst_CsvRow

void tst_Csv() {
  std::istringstream in(
      "first,second,third,fourth\n"
      "a, b,c,\n"
      "1,2,3,4,5,6,7\n"
      "\n"
      );
  Csv csv(in);
  CsvRow row;
  csv >> row;
  auto &header = *row.header();
  TH_EQUAL(header, CsvRow({"first", "second", "third", "fourth"}));
  TH_EQUAL(row, CsvRow({"a", " b", "c", ""}));

  csv >> row;
  TH_EQUAL(row, CsvRow({"1", "2", "3", "4", "5", "6", "7"}));

  csv >> row;
  TH_EQUAL(row, CsvRow({""}));
}
TH_REGISTER(tst_Csv);

void tst_isIn() {
  // an empty vector
  TH_VERIFY(!isIn(std::vector<std::string>{}, ""));

  // non-empty vector
  std::vector<int> vals {1, 3, 1, 2, 5, 7, 8, 5, 5, 7};
  TH_VERIFY(isIn(vals, 3));
  TH_VERIFY(isIn(vals, 5));
  TH_VERIFY(!isIn(vals, 6));

  // array
  std::array<int, 6> arr = {1,2,3,5,3,7};
  TH_VERIFY(isIn(arr, 3));
  TH_VERIFY(!isIn(arr, 6));
}
TH_REGISTER(tst_isIn);

void tst_default_macro_values() {
  TH_EQUAL(FLIT_HOST, "HOST");
  TH_EQUAL(FLIT_COMPILER, "COMPILER");
  TH_EQUAL(FLIT_OPTL, "OPTL");
  TH_EQUAL(FLIT_SWITCHES, "SWITCHES");
  TH_EQUAL(FLIT_NULL, "NULL");
  TH_EQUAL(FLIT_FILENAME, "FILENAME");
}
TH_REGISTER(tst_default_macro_values);

void tst_FlitOptions_toString() {
  flit::FlitOptions opt;
  opt.help = true;
  opt.info = false;
  opt.listTests = false;
  opt.verbose = false;
  opt.tests = {"one", "two", "three"};
  opt.precision = "my precision";
  opt.output = "my output";
  opt.timing = false;
  opt.timingLoops = 100;
  opt.timingRepeats = 2;
  opt.compareMode = true;
  opt.compareGtFile = "MY-GTFILE";
  opt.compareSuffix = "-suffix.csv";
  opt.compareFiles = {"A", "B", "C", "D"};

  TH_EQUAL(opt.toString(),
      "Options:\n"
      "  help:           true\n"
      "  info:           false\n"
      "  verbose:        false\n"
      "  timing:         false\n"
      "  timingLoops:    100\n"
      "  timingRepeats:  2\n"
      "  listTests:      false\n"
      "  precision:      my precision\n"
      "  output:         my output\n"
      "  compareMode:    true\n"
      "  compareGtFile:  MY-GTFILE\n"
      "  compareSuffix:  -suffix.csv\n"
      "  tests:\n"
      "    one\n"
      "    two\n"
      "    three\n"
      "  compareFiles:\n"
      "    A\n"
      "    B\n"
      "    C\n"
      "    D\n"
      );

  // Also test the << operator for streams
  std::ostringstream out;
  out << opt;
  TH_EQUAL(opt.toString(), out.str());
}
TH_REGISTER(tst_FlitOptions_toString);

class A {}; class B {};
namespace std {
  template<> struct hash<A> { size_t operator()(const A&) { return 1234; } };
  template<> struct hash<B> { size_t operator()(const B&) { return 4321; } };
}
void tst_pair_hash() {
  std::pair<A, B> p;
  size_t expected = 0x345678;
  expected = (1000003 * expected) ^ 1234;
  expected = (1000003 * expected) ^ 4321;
  size_t actual = flit::pair_hash<A, B>()(p);
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_pair_hash);

namespace tst_parseArguments {
bool operator==(const flit::FlitOptions& a, const flit::FlitOptions& b) {
  return a.toString() == b.toString();
}
bool operator!=(const flit::FlitOptions& a, const flit::FlitOptions& b) {
  return ! (a == b);
}
void tst_parseArguments_empty() {
  std::vector<const char*> argList = {"progName"};
  flit::FlitOptions expected;
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_empty);

void tst_parseArguments_one_flag() {
  std::vector<const char*> argList = {"progName", "-h"};
  flit::FlitOptions expected;
  expected.help = true;
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_one_flag);

void tst_parseArguments_short_flags() {
  std::vector<const char*> argList = {"progName",
    "-h", "-v", "-t", "-L", "-c", // bool flags
    "-l", "323",
    "-r", "21",
    "-p", "double",
    "-o", "out.txt",
    "-s", "mysuffix.csv",
    "-g", "gtrun.csv",
    "comp1", "comp2", "comp3",
  };
  flit::FlitOptions expected;
  expected.help = true;
  expected.verbose = true;
  expected.timing = false;
  expected.listTests = true;
  expected.compareMode = true;
  expected.timingLoops = 323;
  expected.timingRepeats = 21;
  expected.precision = "double";
  expected.output = "out.txt";
  expected.compareGtFile = "gtrun.csv";
  expected.compareSuffix = "mysuffix.csv";
  expected.compareFiles = {"comp1", "comp2", "comp3"};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_short_flags);

void tst_parseArguments_long_flags() {
  std::vector<const char*> argList = {"progName",
    "--help", "--verbose", "--list-tests", "--compare-mode", "--timing",
    "--timing-loops", "323",
    "--timing-repeats", "21",
    "--precision", "double",
    "--output", "out.txt",
    "--compare-gt", "my-gtrun-output-csv",
    "--suffix", "my-suffix",
    "comp1", "comp2", "comp3",
  };
  flit::FlitOptions expected;
  expected.help = true;
  expected.verbose = true;
  expected.timing = true;
  expected.listTests = true;
  expected.compareMode = true;
  expected.timingLoops = 323;
  expected.timingRepeats = 21;
  expected.precision = "double";
  expected.output = "out.txt";
  expected.compareGtFile = "my-gtrun-output-csv";
  expected.compareSuffix = "my-suffix";
  expected.compareFiles = {"comp1", "comp2", "comp3"};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_long_flags);

void tst_parseArguments_compare_test_names() {
  // tests that the parseArguments does not read the files - keep it simple
  TempFile tmpf;
  tmpf.out << "name,precision,score_hex,resultfile,nanosec\n"
           << "test1,d,0x0,NULL,0\n"
           << "test2,d,0x0,NULL,0\n"
           << "test3,d,0x0,NULL,0";
  tmpf.out.flush();
  std::vector<const char*> argList = {"progName",
    "--compare-mode", tmpf.name.c_str()
  };
  flit::FlitOptions expected;
  expected.compareMode = true;
  expected.timing = false;
  expected.compareFiles = {tmpf.name};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_compare_test_names);

void tst_parseArguments_unrecognized_flag() {
  std::vector<const char*> argList = {"progName", "-T"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
}
TH_REGISTER(tst_parseArguments_unrecognized_flag);

void tst_parseArguments_unknown_precision() {
  std::vector<const char*> argList = {"progName", "--precision", "half"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
}
TH_REGISTER(tst_parseArguments_unknown_precision);

void tst_parseArguments_valid_precisions() {
  std::vector<const char*> argList = {"progName",
    "--precision", "all",
    "--precision", "float",
    "--precision", "double",
    "--precision", "long double",
    "--no-timing",
  };
  flit::FlitOptions expected;
  expected.precision = "long double";
  expected.timing = false;
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseArguments_valid_precisions);

void tst_parseArguments_requires_argument() {
  std::vector<const char*> argList;
  argList = {"progName", "--precision"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
  argList = {"progName", "--timing-loops"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
  argList = {"progName", "--timing-repeats"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
  argList = {"progName", "--output"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
  argList = {"progName", "--compare-gt"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
  argList = {"progName", "--suffix"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);

  // Giving a flag after a parameter option will result in the parameter option
  // assuming the flag is the argument to store.
  argList = {"progName", "--output", "--help"};
  flit::FlitOptions expected;
  expected.output = "--help";
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseArguments_requires_argument);

void tst_parseArguments_expects_integers() {
  std::vector<const char*> argList;
  argList = {"progName", "--timing-loops", "123abc"};
  flit::FlitOptions expected;
  expected.timingLoops = 123;
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(actual, expected);

  argList = {"progName", "--timing-loops", "abc"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);

  argList = {"progName", "--timing-repeats", "abc"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);
}
TH_REGISTER(tst_parseArguments_expects_integers);

struct TestContainerDeleter {
  ~TestContainerDeleter() { flit::getTests().clear(); }
};
void tst_parseArguments_specify_tests() {
  // Setup to empty the getTests() map when the test ends
  // even if an exception is thrown
  TestContainerDeleter deleter;
  (void)deleter;  // suppresses the warning that deleter is not used

  std::vector<const char*> argList = {"progName", "test1", "test2"};
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);

  flit::getTests()["test1"] = nullptr;
  TH_THROWS(flit::parseArguments(argList.size(), argList.data()),
            flit::ParseException);

  flit::getTests()["test2"] = nullptr;
  flit::FlitOptions expected;
  expected.tests = {"test1", "test2"};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseArguments_specify_tests);

void tst_parseArguments_all_tests_expand() {
  // Setup to empty the getTests() map when the test ends
  // even if an exception is thrown
  TestContainerDeleter deleter;
  (void)deleter;  // suppresses the warning that deleter is not used

  flit::getTests()["test1"] = nullptr;
  flit::getTests()["test2"] = nullptr;
  flit::getTests()["test3"] = nullptr;

  // even if tests are provided, if "all" is there, just have each test once
  std::vector<const char*> argList;
  argList = {"progName", "test3", "all"};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test1"));
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test2"));
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test3"));

  // if no tests are provided, then use all tests
  argList = {"progName"};
  actual = flit::parseArguments(argList.size(), argList.data());
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test1"));
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test2"));
  TH_EQUAL(1, std::count(actual.tests.begin(), actual.tests.end(), "test3"));
}
TH_REGISTER(tst_parseArguments_all_tests_expand);

void tst_parseArguments_specify_test_more_than_once() {
  // Setup to empty the getTests() map when the test ends
  // even if an exception is thrown
  TestContainerDeleter deleter;
  (void)deleter;  // suppresses the warning that deleter is not used

  flit::getTests()["test1"] = nullptr;
  flit::getTests()["test2"] = nullptr;
  flit::getTests()["test3"] = nullptr;

  std::vector<const char*> argList = {"progName", "test2", "test3", "test2"};
  auto actual = flit::parseArguments(argList.size(), argList.data());
  decltype(actual.tests) expected_tests {"test2", "test3", "test2"};
  TH_EQUAL(actual.tests, expected_tests);
}
TH_REGISTER(tst_parseArguments_specify_test_more_than_once);

} // end of namespace tst_parseArguments

void tst_usage() {
  const auto usage = flit::usage("progName");
  auto usage_contains = [&usage](const std::string &x) {
    return std::string::npos != usage.find(x);
  };

  // only test that the usage has some key elements.  Other than this, it
  // should be manually inspected.
  TH_VERIFY(usage_contains("Usage:\n"));
  TH_VERIFY(usage_contains("progName [options] [<test> ...]\n"));
  TH_VERIFY(usage_contains("progName --compare-mode <csvfile> [<csvfile> ...]\n"));
  TH_VERIFY(usage_contains("Description:\n"));
  TH_VERIFY(usage_contains("Options:\n"));
  TH_VERIFY(usage_contains("-h, --help"));
  TH_VERIFY(usage_contains("-L, --list-tests"));
  TH_VERIFY(usage_contains("-v, --verbose"));
  TH_VERIFY(usage_contains("-t, --timing"));
  TH_VERIFY(usage_contains("--no-timing"));
  TH_VERIFY(usage_contains("-r REPEATS, --timing-repeats REPEATS"));
  TH_VERIFY(usage_contains("-o OUTFILE, --output OUTFILE"));
  TH_VERIFY(usage_contains("-c, --compare-mode"));
  TH_VERIFY(usage_contains("-g GT_RESULTS, --compare-gt GT_RESULTS"));
  TH_VERIFY(usage_contains("-s SUFFIX, --suffix SUFFIX"));
  TH_VERIFY(usage_contains("-p PRECISION, --precision PRECISION"));
  TH_VERIFY(usage_contains("'float'"));
  TH_VERIFY(usage_contains("'double'"));
  TH_VERIFY(usage_contains("'long double'"));
  TH_VERIFY(usage_contains("'all'"));
}
TH_REGISTER(tst_usage);

void tst_readFile_exists() {
  TempFile tmpf;
  std::string contents =
    "This is the sequence of characters and lines\n"
    "that I want to check that the readFile()\n"
    "can return.\n"
    "\n"
    "\n"
    "You okay with that?";
  tmpf.out << contents;
  tmpf.out.flush();

  TH_EQUAL(contents, flit::readFile(tmpf.name));
}
TH_REGISTER(tst_readFile_exists);

void tst_readFile_doesnt_exist() {
  TH_THROWS(flit::readFile("/this/file/should/not/exist"),
            std::ios_base::failure);
}
TH_REGISTER(tst_readFile_doesnt_exist);

namespace flit {
  // Note: if you do not put this in the flit namespace, then I cannot do a
  //   direct comparison of vectors of TestResult objects -- it would result in a
  //   compiler error.
  bool operator==(const TestResult a, const TestResult b) {
    return
      a.name() == b.name() &&
      a.precision() == b.precision() &&
      a.result() == b.result() &&
      a.nanosecs() == b.nanosecs() &&
      a.comparison() == b.comparison() &&
      a.is_comparison_null() == b.is_comparison_null() &&
      a.resultfile() == b.resultfile();
  }
}
void tst_parseResults() {
  std::istringstream in(
      "name,precision,score_hex,resultfile,nanosec\n"
      "Mike,double,0x00000000000000000000,output.txt,149293\n"
      "Brady,long double,0x3fff8000000000000000,NULL,-1\n"
      "Julia,float,NULL,test-output.txt,498531\n"
      );
  auto results = flit::parseResults(in);

  decltype(results) expected;
  expected.emplace_back("Mike", "double", flit::Variant(0.0), 149293, "");
  expected.emplace_back("Brady", "long double", flit::Variant(1.0), -1, "");
  expected.emplace_back("Julia", "float", flit::Variant(), 498531, "test-output.txt");

  TH_EQUAL(results, expected);
}
TH_REGISTER(tst_parseResults);

void tst_parseResults_invalid_format() {
  // header does not contain the correct columns
  std::istringstream in;
  in.str("hello,there\n"
         "Mike,double\n"
         "Brady,float\n");
  TH_THROWS(flit::parseResults(in), std::invalid_argument);

  // empty row
  in.str("name,precision,score_hex,resultfile,nanosec\n"
         "\n");
  TH_THROWS(flit::parseResults(in), std::out_of_range);

  // non-integer nanosec
  in.str("name,precision,score_hex,resultfile,nanosec\n"
         "Mike,double,0x0,NULL,bob\n");
  TH_THROWS(flit::parseResults(in), std::invalid_argument);

  // non-integer score
  in.str("name,precision,score_hex,resultfile,nanosec\n"
         "Mike,double,giraffe,NULL,323\n");
  TH_THROWS(flit::parseResults(in), std::invalid_argument);

  // doesn't end in a newline.  Make sure it doesn't throw
  in.str("name,precision,score_hex,resultfile,nanosec\n"
         "Mike,double,0x0,NULL,323");
  auto actual = flit::parseResults(in);
  decltype(actual) expected;
  expected.emplace_back("Mike", "double", flit::Variant(0.0), 323, "");
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseResults_invalid_format);

void tst_parseMetadata() {
  // This just parses the first row of data.  If there is no row, then an empty
  // map is returned.  Only the following columns are used in retrieving
  // metadata
  // - host
  // - compiler
  // - optl
  // - switches
  // - file
  // extra columns are not used.

  std::istringstream in;
  in.str(
      "unused,host,compiler,optl,switches,file\n"
      "hello,my host,g++,-O3,,executable file,ignored,ignored again\n"
      "ignored,ignored,ignored,ignored,ignored,ignored,ignored,ignored,ignored\n"
      );
  auto metadata = flit::parseMetadata(in);
  decltype(metadata) expected;
  expected["host"]     = "my host";
  expected["compiler"] = "g++";
  expected["optl"]     = "-O3";
  expected["switches"] = "";
  expected["file"]     = "executable file";
  TH_EQUAL(metadata, expected);
  expected.clear();

  // no rows should be fine
  in.str(
      "unused,host,compiler,optl,switches,file\n"
      );
  metadata = flit::parseMetadata(in);
  TH_EQUAL(metadata, expected);

  // not even a header row should be fine
  in.str("");
  metadata = flit::parseMetadata(in);
  TH_EQUAL(metadata, expected);
}
TH_REGISTER(tst_parseMetadata);

void tst_removeIdxFromName() {
  TH_EQUAL(flit::removeIdxFromName("hello there"), "hello there");
  TH_EQUAL(flit::removeIdxFromName("hello there_idx0"), "hello there");
  TH_THROWS(flit::removeIdxFromName("hello there_idxa"), std::invalid_argument);
}
TH_REGISTER(tst_removeIdxFromName);

void tst_calculateMissingComparisons_empty() {
  flit::FlitOptions options;
  std::vector<std::string> expected;
  auto actual = calculateMissingComparisons(options);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_calculateMissingComparisons_empty);

void tst_calculateMissingComparisons_noGtFile() {
  flit::FlitOptions options;
  options.compareMode = true;
  TempFile tmpf;
  tmpf.out << "name,precision,score_hex,resultfile,nanosec\n"
           << "test1,d,0x0,NULL,0\n"
           << "test2,d,0x0,NULL,0\n"
           << "test3,d,0x0,NULL,0";
  tmpf.out.flush();
  options.compareFiles = {tmpf.name};
  std::vector<std::string> expected = {"test1", "test2", "test3"};
  auto actual = calculateMissingComparisons(options);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_calculateMissingComparisons_noGtFile);

void tst_calculateMissingComparisons_withGtFile() {
  TempFile compf1;
  compf1.out << "name,precision,score_hex,resultfile,nanosec\n"
             << "test1,d,0x0,NULL,0\n"
             << "test2,d,0x0,NULL,0\n"
             << "test3,d,0x0,NULL,0\n";
  compf1.out.flush();
  TempFile compf2;
  compf2.out << "name,precision,score_hex,resultfile,nanosec\n"
             << "test2,d,0x0,NULL,0\n"
             << "test4,d,0x0,NULL,0\n"
             << "test6,d,0x0,NULL,0\n";
  compf2.out.flush();
  TempFile gtf;
  gtf.out << "name,precision,score_hex,resultfile,nanosec\n"
          << "test1,d,0x0,NULL,0\n"
          << "test2,d,0x0,NULL,0\n"
          << "test5,d,0x0,NULL,0\n";
  gtf.out.flush();
  flit::FlitOptions options;
  options.compareMode = true;
  options.compareFiles = {compf1.name, compf2.name};
  options.compareGtFile = gtf.name;
  std::vector<std::string> expected = {"test3", "test4", "test6"};
  auto actual = calculateMissingComparisons(options);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_calculateMissingComparisons_withGtFile);
