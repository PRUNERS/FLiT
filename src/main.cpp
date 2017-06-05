// This is the main implementation, handling cmd line params and
// running the tests!

#include <cstring>
#include <typeinfo>
#include <chrono>
#include <type_traits>

#include "TestBase.hpp"
#include "QFPHelpers.hpp"

#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.hpp"
using namespace CUHelpers;
#endif


using namespace QFPHelpers;
using namespace QFPTest;

//this sets the test parameters
namespace {

bool
loadBoolFromEnv(const std::string &var){
  return std::getenv(var.c_str()) != nullptr;
}

const std::string
loadStringFromEnv(const std::string &var,
                  const std::string &defVal) {
  if(std::getenv(var.c_str()) == nullptr) {
    return defVal;
  } else {
    return std::getenv(var.c_str());
  }
}

int
loadIntFromEnv(const std::string &var, int defVal){
  const char* res = std::getenv(var.c_str());
  if(res == nullptr || std::strlen(res) == 0) {
    return defVal;
  } else {
    return std::atoi(res);
  }
}

std::string const TEST = loadStringFromEnv("TEST", "all");
std::string const PRECISION = loadStringFromEnv("PRECISION", "all");
bool const doOne = TEST != "all";
bool const GetTime = loadBoolFromEnv("DO_TIMINGS");
int const TimingLoops = loadIntFromEnv("TIMING_LOOPS", 10);


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
                             QFPTest::ResultType& totScores) {
  using namespace std::chrono;
    
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();
  auto scores = test->run(ip, GetTime, TimingLoops);
  totScores.insert(scores.begin(), scores.end());
  info_stream.flushout();
}

} //namespace

int main(int argc, char* argv[]) {

  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  if((TEST == "all") != (PRECISION == "all")){ //all or one
    std::cerr << argv[0] << " must be run with one or all tests selected."
              << std::endl;
    return 1;
  }

  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

#ifdef __CUDA__
  CUHelpers::initDeviceData();
#endif

  // if(NO_WATCHS != "true"){
  //   QFPTest::setWatching();
  // }

  QFPTest::ResultType scores;
  if(doOne) {
    auto factory = getTests()[TEST];
    std::string precision(PRECISION);
    if(precision == "f") {
      runTestWithDefaultInput<float>(factory, scores);
    } else if (precision == "d") {
      runTestWithDefaultInput<double>(factory, scores);
    } else {
      runTestWithDefaultInput<long double>(factory, scores);
    }
  } else {
    for(auto& t : getTests()) {
      auto factory = t.second;
      runTestWithDefaultInput<float>(factory, scores);
      runTestWithDefaultInput<double>(factory, scores);
      runTestWithDefaultInput<long double>(factory, scores);
    }
  }
#if defined(__CUDA__) && !defined(__CPUKERNEL__)
  cudaDeviceSynchronize();
#endif

  outputResults(scores);
  return 0;
}
