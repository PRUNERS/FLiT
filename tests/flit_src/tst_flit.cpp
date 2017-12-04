#include "test_harness.h"

#include "flit.h"
#include "flit.cpp"

#include "TestBase.h"   // for operator<<(flit::TestResult ...)

#include <algorithm>
#include <array>
#include <sstream>
#include <vector>

#include <cstdio>

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
  opt.listTests = false;
  opt.verbose = false;
  opt.tests = {"one", "two", "three"};
  opt.precision = "my precision";
  opt.output = "my output";
  opt.timing = false;
  opt.timingLoops = 100;
  opt.timingRepeats = 2;
  opt.compareMode = true;
  opt.compareFiles = {"A", "B", "C", "D"};

  TH_EQUAL(opt.toString(),
      "Options:\n"
      "  help:           true\n"
      "  verbose:        false\n"
      "  timing:         false\n"
      "  timingLoops:    100\n"
      "  timingRepeats:  2\n"
      "  listTests:      false\n"
      "  precision:      my precision\n"
      "  output:         my output\n"
      "  compareMode:    true\n"
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
  const char* argList[1] = {"progName"};
  flit::FlitOptions expected;
  auto actual = flit::parseArguments(1, argList);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_empty);

void tst_parseArguments_one_flag() {
  const char* argList[2] = {"progName", "-h"};
  flit::FlitOptions expected;
  expected.help = true;
  auto actual = flit::parseArguments(2, argList);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_one_flag);

void tst_parseArguments_short_flags() {
  const char* argList[17] = {"progName",
    "-h", "-v", "-t", "-L", "-c", // bool flags
    "-l", "323",
    "-r", "21",
    "-p", "double",
    "-o", "out.txt",
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
  expected.compareFiles = {"comp1", "comp2", "comp3"};
  auto actual = flit::parseArguments(17, argList);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_short_flags);

void tst_parseArguments_long_flags() {
  const char* argList[17] = {"progName",
    "--help", "--verbose", "--timing", "--list-tests", "--compare-mode",
    "--timing-loops", "323",
    "--timing-repeats", "21",
    "--precision", "double",
    "--output", "out.txt",
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
  expected.compareFiles = {"comp1", "comp2", "comp3"};
  auto actual = flit::parseArguments(17, argList);
  TH_EQUAL(expected, actual);
}
TH_REGISTER(tst_parseArguments_long_flags);

void tst_parseArguments_unrecognized_flag() {
  const char* argList[2] = {"progName", "-T"};
  TH_THROWS(flit::parseArguments(2, argList), flit::ParseException);
}
TH_REGISTER(tst_parseArguments_unrecognized_flag);

void tst_parseArguments_unknown_precision() {
  const char* argList[3] = {"progName", "--precision", "half"};
  TH_THROWS(flit::parseArguments(2, argList), flit::ParseException);
}
TH_REGISTER(tst_parseArguments_unknown_precision);

void tst_parseArguments_valid_precisions() {
  const char* argList[10] = {"progName",
    "--precision", "all",
    "--precision", "float",
    "--precision", "double",
    "--precision", "long double",
    "--no-timing",
  };
  flit::FlitOptions expected;
  expected.precision = "long double";
  expected.timing = false;
  auto actual = flit::parseArguments(10, argList);
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseArguments_valid_precisions);

void tst_parseArguments_requires_argument() {
  const char* argList1[2] = {"progName", "--precision"};
  TH_THROWS(flit::parseArguments(2, argList1), flit::ParseException);
  const char* argList2[2] = {"progName", "--timing-loops"};
  TH_THROWS(flit::parseArguments(2, argList2), flit::ParseException);
  const char* argList3[2] = {"progName", "--timing-repeats"};
  TH_THROWS(flit::parseArguments(2, argList3), flit::ParseException);
  const char* argList4[2] = {"progName", "--output"};
  TH_THROWS(flit::parseArguments(2, argList4), flit::ParseException);

  // Giving a flag after a parameter option will result in the parameter option
  // assuming the flag is the argument to store.
  const char* argList5[3] = {"progName", "--output", "--help"};
  flit::FlitOptions expected;
  expected.output = "--help";
  auto actual = flit::parseArguments(3, argList5);
  TH_EQUAL(actual, expected);
}
TH_REGISTER(tst_parseArguments_requires_argument);

void tst_parseArguments_expects_integers() {
  const char* argList1[3] = {"progName", "--timing-loops", "123abc"};
  flit::FlitOptions expected;
  expected.timingLoops = 123;
  auto actual = flit::parseArguments(3, argList1);
  TH_EQUAL(actual, expected);
  
  const char* argList2[3] = {"progName", "--timing-loops", "abc"};
  TH_THROWS(flit::parseArguments(3, argList2), flit::ParseException);
  
  const char* argList3[3] = {"progName", "--timing-repeats", "abc"};
  TH_THROWS(flit::parseArguments(3, argList3), flit::ParseException);
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

  const char* argList[3] = {"progName", "test1", "test2"};
  TH_THROWS(flit::parseArguments(3, argList), flit::ParseException);

  flit::getTests()["test1"] = nullptr;
  TH_THROWS(flit::parseArguments(3, argList), flit::ParseException);

  flit::getTests()["test2"] = nullptr;
  flit::FlitOptions expected;
  expected.tests = {"test1", "test2"};
  auto actual = flit::parseArguments(3, argList);
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
  const char* argList1[3] = {"progName", "test3", "all"};
  auto actual1 = flit::parseArguments(3, argList1);
  TH_EQUAL(1, std::count(actual1.tests.begin(), actual1.tests.end(), "test1"));
  TH_EQUAL(1, std::count(actual1.tests.begin(), actual1.tests.end(), "test2"));
  TH_EQUAL(1, std::count(actual1.tests.begin(), actual1.tests.end(), "test3"));

  // if no tests are provided, then use all tests
  const char* argList2[1] = {"progName"};
  auto actual2 = flit::parseArguments(1, argList2);
  TH_EQUAL(1, std::count(actual2.tests.begin(), actual2.tests.end(), "test1"));
  TH_EQUAL(1, std::count(actual2.tests.begin(), actual2.tests.end(), "test2"));
  TH_EQUAL(1, std::count(actual2.tests.begin(), actual2.tests.end(), "test3"));
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

  const char* argList[4] = {"progName", "test2", "test3", "test2"};
  auto actual = flit::parseArguments(4, argList);
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
  TH_VERIFY(usage_contains("-p PRECISION, --precision PRECISION"));
  TH_VERIFY(usage_contains("'float'"));
  TH_VERIFY(usage_contains("'double'"));
  TH_VERIFY(usage_contains("'long double'"));
  TH_VERIFY(usage_contains("'all'"));
}
TH_REGISTER(tst_usage);

void tst_readFile_exists() {
  struct TmpFile {
    std::ofstream out;
    std::string fname;

    TmpFile() {
      char fname_buf[L_tmpnam];
      auto ptr = std::tmpnam(fname_buf); // gives a warning, but I'm not worried

      fname = fname_buf;
      fname += "-tst_flit.in";           // this makes the danger much less likely
      out.exceptions(std::ios::failbit);
      out.open(fname);
    }

    ~TmpFile() {
      out.close();
      std::remove(fname.c_str());
    }
  };
  TmpFile tmp;
  std::string contents =
    "This is the sequence of characters and lines\n"
    "that I want to check that the readFile()\n"
    "can return.\n"
    "\n"
    "\n"
    "You okay with that?";
  tmp.out << contents;
  tmp.out.flush();

  TH_EQUAL(contents, flit::readFile(tmp.fname));
}
TH_REGISTER(tst_readFile_exists);

void tst_readFile_doesnt_exist() {
  TH_THROWS(flit::readFile("/this/file/should/not/exist"),
            std::system_error);
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
      "name,precision,score,resultfile,nanosec\n"
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
  in.str("name,precision,score,resultfile,nanosec\n"
         "\n");
  TH_THROWS(flit::parseResults(in), std::out_of_range);

  // non-integer nanosec
  in.str("name,precision,score,resultfile,nanosec\n"
         "Mike,double,0x0,NULL,bob\n");
  TH_THROWS(flit::parseResults(in), std::invalid_argument);

  // non-integer score
  in.str("name,precision,score,resultfile,nanosec\n"
         "Mike,double,giraffe,NULL,323\n");
  TH_THROWS(flit::parseResults(in), std::invalid_argument);

  // doesn't end in a newline.  Make sure it doesn't throw
  in.str("name,precision,score,resultfile,nanosec\n"
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
