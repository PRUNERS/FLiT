// This is the main implementation, handling cmd line params and
// running the tests!

#include <cstring>
#include <typeinfo>

#include "testBase.hpp"
#include "QFPHelpers.hpp"

#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.hpp"
using namespace CUHelpers;
#endif


using namespace QFPHelpers;
using namespace QFPTest;

void
loadStringFromEnv(std::string &dest, const std::string &var,
                  const std::string &defVal) {
  if(std::getenv(var.c_str()) == nullptr) {
    dest = defVal;
  } else {
    dest = std::getenv(var.c_str());
  }
}

void
loadIntFromEnv(int &dest, const std::string &var, int defVal){
  const char* res = std::getenv(var.c_str());
  if(res == nullptr || std::strlen(res) == 0) {
    dest = defVal;
  } else {
    dest = std::atoi(res);
  }
}

void
outputResults(const QFPTest::ResultType& scores){
  for(const auto& i: scores){
    std::cout
      << "HOST,SWITCHES,COMPILER,"
      << i.first.second << ",us,"
      << i.second.first << ","
      << as_int(i.second.first) << ","
      << i.second.second << ","
      << as_int(i.second.second) << ","
      << i.first.first << ","
      << "FILENAME"
      << std::endl;
  }
}

template <typename F>
void runTestWithDefaultInput(QFPTest::TestFactory* factory,
                             QFPTest::ResultType& totScores) {
  auto test = factory->get<F>();
  auto ip = test->getDefaultInput();

  auto scores = test->run(ip);
  totScores.insert(scores.begin(), scores.end());
  info_stream.flushout();
}

int main(int argc, char* argv[]) {

  std::string NO_WATCHS;
  loadStringFromEnv(NO_WATCHS, "NO_WATCH", "true");
  std::string sfx;
  if(NO_WATCHS != "true")
    sfx = argv[0][std::strlen(argv[0]) - 1]; //to diff btw inf1 & inf2
  else
    sfx = "";
  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  std::string TEST;
  loadStringFromEnv(TEST, std::string("TEST") + sfx, "all");
  std::string PRECISION;
  loadStringFromEnv(PRECISION, std::string("PRECISION") + sfx, "all");
  bool doOne = TEST != "all";
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

  if(NO_WATCHS != "true"){
    QFPTest::setWatching();
  }

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

#ifdef __CUDA__
  cudaDeviceSynchronize();
#endif

  outputResults(scores);
}


