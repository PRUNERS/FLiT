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

#include "helper.h"
#include "groundtruth.h"

#include <flit.h>

#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include <cmath>

#include <dlfcn.h>    // For dlopen(), dlsym() and dlclose()

//TESTRUN_DEFINE(distribution, 3, RandomFloatType::Positive)

template<typename T> const char* testbedSymbolName();
template<> const char* testbedSymbolName<float>() { return "runTestbed_float"; }
template<> const char* testbedSymbolName<double>() { return "runTestbed_double"; }
template<> const char* testbedSymbolName<long double>() { return "runTestbed_longdouble"; }

template<typename T> long double
runTestbed(const std::string &testName, const std::vector<T> &inputs) {
  const char* lib = "./testbed.so";
  const char* symb = testbedSymbolName<T>();

  using TestBedFnc = long double(const std::string&, const std::vector<T>&);
  auto handle = dlopen(lib,
      RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  auto testbed = reinterpret_cast<TestBedFnc*>(dlsym(handle, symb));
  if (testbed == nullptr) {
    std::cerr << "Error: could not find symbol " << symb << " from " << lib << std::endl;
    std::exit(1);
  }
  auto retval = testbed(testName, inputs);

  dlclose(handle);

  return retval;
}


template<typename T>
void runTest(std::string testName, uint divergentCount, uint maxTries,
             RandType rType = RandType::UniformFP) {
  const std::function<T()> randGen = [rType]() {
    return generateRandomFloat<T>(RandomFloatType::Positive, rType);
  };

  std::cout << testName << "(" << typeid(T).name() << "): "
            << "looking for " << divergentCount << " divergences, "
            << "max_tries = " << maxTries
            << std::endl;

  uint mismatchIdx;
  std::vector<std::vector<T>> mismatches;
  for (mismatchIdx = 0;
       mismatches.size() < divergentCount && mismatchIdx < maxTries;
       mismatchIdx++)
  {
    auto truthRun = runGroundtruth(testName, randGen);
    auto inputs = truthRun.first;
    auto truthval = truthRun.second;
    auto testval = runTestbed(testName, inputs);
    if (truthval != testval && !(std::isnan(truthval) && std::isnan(testval))) {
      mismatches.push_back(inputs);

      std::cout << testName << ":   Divergent outputs #" << mismatches.size() << std::endl;
      printTestVal(testName, truthval);
      printTestVal(testName, testval);
      std::cout << testName << ":   Divergent inputs #" << mismatches.size() << std::endl;
      for (auto& val : inputs) {
        printTestVal(testName, val);
      }
    }
  }
  std::cout << testName << "(" << typeid(T).name() << "): "
            << mismatches.size() << " Diverging inputs found "
            << "after " << mismatchIdx << " iterations:"
            << std::endl
            << std::endl;
}

void runAllPrecisions(const std::string testName, uint divergentCount,
                      uint maxTries, RandType rType = RandType::UniformFP) {
  runTest<float>(testName, divergentCount, maxTries, rType);
  runTest<double>(testName, divergentCount, maxTries, rType);
  runTest<long double>(testName, divergentCount, maxTries, rType);
}

std::vector<std::string> getTestNames() {
  std::vector<std::string> retval;
  for (auto entry : flit::getTests()) {
    retval.push_back(entry.first);
  }
  return retval;
}

/// An RAII class for replacing a stream buffer and restoring in destruction
class BufSwap {
public:
  BufSwap(std::ostream& stream, std::streambuf* newbuf)
    : _stream(stream), _oldbuf(stream.rdbuf()), _newbuf(newbuf)
  { _stream.rdbuf(_newbuf); }

  ~BufSwap() { _stream.rdbuf(_oldbuf); }

private:
  std::ostream& _stream;
  std::streambuf* _oldbuf;
  std::streambuf* _newbuf;
};

/// All options from command-line parsing are stored here
struct ProgramArguments {
  enum class PrecisionType {
    Float,
    Double,
    LongDouble,
    All,
  };
  PrecisionType type = PrecisionType::Float;
  RandType randType = RandType::UniformFP;
  uint maxTries = 1000;
  bool outputToFile = false;
  std::string outFilename;
  uint divergentCount = 10;
  std::vector<std::string> tests;
};
ProgramArguments parseArgs(int argCount, char* argList[]);

int main(int argCount, char* argList[]) {
  auto parsedArgs = parseArgs(argCount, argList);

  auto runner = runAllPrecisions;
  switch(parsedArgs.type) {
    case ProgramArguments::PrecisionType::Float:
      runner = runTest<float>;
      break;
    case ProgramArguments::PrecisionType::Double:
      runner = runTest<double>;
      break;
    case ProgramArguments::PrecisionType::LongDouble:
      runner = runTest<long double>;
      break;
    case ProgramArguments::PrecisionType::All:
      runner = runAllPrecisions;
      break;
  }

  // Set the console output to a file if specified
  std::unique_ptr<std::ofstream> out(nullptr);
  std::unique_ptr<BufSwap> swapper(nullptr);
  if (parsedArgs.outputToFile) {
    out.reset(new std::ofstream(parsedArgs.outFilename));
    swapper.reset(new BufSwap(std::cout, out->rdbuf()));
  }

  for (auto test : parsedArgs.tests) {
    runner(test, parsedArgs.divergentCount, parsedArgs.maxTries,
           parsedArgs.randType);
  }

  return 0;
}

ProgramArguments parseArgs(int argCount, char* argList[]) {
  const char* USAGE_INFO = "\n"
    "Usage:\n"
    "  inputGen [-h]\n"
    "  inputGen --list-tests\n"
    "  inputGen [-f|-d|-e|-a] [-m MAX_TRIES] [-i NUM_DIVERGENT_INPUTS] [-o FILENAME]\n"
    "           [-r RAND_TYPE] (--test TEST_NAME ... | --all-tests)\n"
    "\n"
    "Description:\n"
    "  Runs the particular tests under the optimization given in compilation.  The\n"
    "  test input is randomly generated floating point numbers.  It runs until the\n"
    "  number of desired divergences are discovered between the unoptimized code\n"
    "  and the optimized code.\n"
    "\n"
    "Required Arguments:\n"
    "  One of the following must be used since we need at least one test to run.\n"
    "  The --test option can be specified more than once to specify more than one\n"
    "  test.\n"
    "\n"
    "  --test TEST_NAME\n"
    "      The test to run.  For a list of choices, see --list-tests.\n"
    "\n"
    "  --all-tests\n"
    "      Run all tests that are returned from --list-tests.\n"
    "\n"
    "Optional Arguments:\n"
    "\n"
    "  -f, --float\n"
    "      Run each test only with float precision.  This is the default.\n"
    "\n"
    "  -d, --double\n"
    "      Run each test only with double precision.\n"
    "\n"
    "  -e, --long-double\n"
    "      Run each test only with extended precision, also called long double.\n"
    "\n"
    "  -a, --all-types\n"
    "      Run each test with all precisions.\n"
    "\n"
    "  -i NUM_DIVERGENT_INPUTS, --divergent-input-count NUM_DIVERGEN_INPUTS\n"
    "      How many divergent inputs to attempt to identify and report.  The\n"
    "      default value is 10.\n"
    "\n"
    "  -m MAX_TRIES, --max-tries MAX_TRIES\n"
    "      How many tries before giving up on search for the specified numer of\n"
    "      divergent inputs.  The default value is 1,000.\n"
    "\n"
    "  -o FILENAME, --output FILENAME\n"
    "      Where to output the report.  By default it outputs to stdout.\n"
    "\n"
    "  -r RAND_TYPE, --rand-type RAND_TYPE\n"
    "      The type of random number distribution to use.  Choices are:\n"
    "      - fp: Uniform over the floating-point number space\n"
    "      - real: Uniform over the real number line, then projected to\n"
    "        floating-point space\n"
    "      The default is \"fp\".\n"
    "\n"
    ;

  std::vector<std::string> args;
  for (decltype(argCount) i = 1; i < argCount; i++) {
    args.push_back(argList[i]);
  }

  ProgramArguments parsedArgs;

  auto is_in = [](std::vector<std::string> vec, std::string val) {
    return std::any_of(vec.begin(), vec.end(),
        [&val](std::string it) { return it == val; });
  };

  // Look for help
  if (is_in(args, "-h") || is_in(args, "--help")) {
    std::cout << USAGE_INFO;
    std::exit(0);
  }

  // Look for --list-tests
  if (is_in(args, "--list-tests")) {
    for (auto test : getTestNames()) {
      std::cout << test << std::endl;
    }
    std::exit(0);
  }

  // Handle any other argument
  for (decltype(args.size()) i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    if (arg == "-f" || arg == "--float") {
      parsedArgs.type = ProgramArguments::PrecisionType::Float;
    } else if (arg == "-d" || arg == "--double") {
      parsedArgs.type = ProgramArguments::PrecisionType::Double;
    } else if (arg == "-e" || arg == "--long-double") {
      parsedArgs.type = ProgramArguments::PrecisionType::LongDouble;
    } else if (arg == "-a" || arg == "--all-types") {
      parsedArgs.type = ProgramArguments::PrecisionType::All;
    } else if (arg == "-m" || arg == "--max-tries") {
      auto& nextArg = args[++i];
      parsedArgs.maxTries = static_cast<uint>(std::stoul(nextArg));
    } else if (arg == "-o" || arg == "--output") {
      auto& nextArg = args[++i];
      parsedArgs.outputToFile = true;
      parsedArgs.outFilename = nextArg;
    } else if (arg == "-i" || arg == "--divergent-input-count") {
      auto& nextArg = args[++i];
      parsedArgs.divergentCount = static_cast<uint>(std::stoul(nextArg));
    } else if (arg == "--test") {
      auto& nextArg = args[++i];
      parsedArgs.tests.push_back(nextArg);
    } else if (arg == "--all-tests") {
      auto allTests = getTestNames();
      parsedArgs.tests.insert(parsedArgs.tests.end(), allTests.begin(), allTests.end());
    } else if (arg == "-r" || arg == "--rand-type") {
      auto& nextArg = args[++i];
      if (nextArg == "fp") {
        parsedArgs.randType = RandType::UniformFP;
      } else if (nextArg == "real") {
        parsedArgs.randType = RandType::UniformReals;
      } else {
        std::cerr << "Error: unrecognized rand-type: " << nextArg << std::endl;
        std::cerr << "  Use the --help option for more information" << std::endl;
        exit(1);
      }
    } else {
      std::cerr << "Error: unrecognized argument: " << arg << std::endl;
      std::cerr << "  Use the --help option for more information" << std::endl;
      exit(1);
    }
  }

  if (parsedArgs.tests.size() == 0) {
    std::cerr << "Error: you need to specify at least one test" << std::endl;
    std::cerr << "  Use the --help option for more information" << std::endl;
    std::exit(1);
  }

  return parsedArgs;
}
