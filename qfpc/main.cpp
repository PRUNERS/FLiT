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
using namespace QFPHelpers::FPHelpers;
using namespace QFPTest;

int
getPrecID(std::string s){
  if(s == "f") return 0;
  if(s == "d") return 1;
  return 2; //long double
}

void
loadStringFromEnv(std::string &dest, std::string var, std::string defVal){
  // std::cout << "in lsfe, var is: " << var << std::endl;
  if(std::getenv(var.c_str()) == NULL){
    dest = defVal;
  }else{
    dest = std::getenv(var.c_str());
  }
  // std::cout << "env is: " << dest << std::endl;
}

void
loadIntFromEnv(int &dest, std::string var, int defVal){
  const char* res = std::getenv(var.c_str());
  if(res == NULL || std::strlen(res) == 0){
    dest = defVal;
  }else{
    dest = std::atoi(std::getenv(var.c_str()));
  }
}

void
outputResults(const QFPTest::resultType& scores){
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

int
main(int argc, char* argv[]){

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
    std::cerr << argv[0] << " must be ran with one or all tests selected."
              << std::endl;
    return 0;
  }

  size_t iters = 200;
  size_t dim = 16;
  size_t ulp_inc = 1;
  float min = -6.0;
  float max = 6.0;

  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

#ifdef __CUDA__
  CUHelpers::initDeviceData();
#endif

  QFPTest::resultType scores;

  scores.clear();

  if(NO_WATCHS != "true"){
    QFPTest::setWatching();
  }

  //singleton
  if(doOne){
    QFPTest::testInput ip{iters, dim, ulp_inc, min, max};
    auto plist = TestBase::getTests()[TEST]->create();
    auto score = (*plist[getPrecID(PRECISION)])(ip);
    for(auto& p: plist) delete p;
    scores.insert(score.begin(), score.end());
    outputResults(scores);
  }else{

    QFPTest::testInput ip{iters, dim, ulp_inc, min, max};
    scores.clear();
    for(auto& t : TestBase::getTests()){
      auto plist = t.second->create();
      for(auto pt : plist){
        auto score = (*pt)(ip);
        scores.insert(score.begin(), score.end());
      }
    }
    outputResults(scores);
  }
#if defined(__CUDA__) && !defined(__CPUKERNEL__)
  cudaDeviceSynchronize();
#endif
}


