// This is the main implementation, handling cmd line params and
// running the tests!

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include "flit.h"

#include "QFPHelpers.hpp"
#include "TestBase.hpp"

void outputResults(const flit::ResultType& scores){
  using flit::operator<<;
  using flit::as_int;
  for(const auto& i: scores){
    std::cout
      << "HOST,SWITCHES,OPTL,COMPILER,"
      << i.first.second << ",us,"             //sort
      << i.second.first.first << ","          //score0d
      << as_int(i.second.first.first) << ","  //score0
      << i.second.first.second << ","         //score1d
      << as_int(i.second.first.second) << "," //score1
      << i.first.first << ","                 //name
      << i.second.second << ","               //nanoseconds
      << "FILENAME"                           //filename
      << std::endl;
  }
}

std::string FlitOptions::toString() {
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
  auto allowedTests = getKeys(flit::getTests());
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
    options.tests = getKeys(flit::getTests());
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

