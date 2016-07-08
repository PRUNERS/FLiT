// This is the main implementation, handling cmd line params and
// running the tests!

#include <cstring>
#include <typeinfo>
#include <future>
#include <chrono>
#include <list>

#include "testBase.h"
#include "QFPHelpers.h"


using namespace QFPHelpers;
using namespace QFPTest;

// typedef std::map<int,int> score_t;
// typedef std::map<std::pair<std::string, std::string>,
//                  std::pair<long double, long double>> score_t;

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
    std::cout << "HOST,SWITCHES,COMPILER," << i.first.second << ",us," << i.second.first
	      << "," << FPWrap<long double>(i.second.first) << ","
	      << i.second.second << "," << FPWrap<long double>(i.second.second) << ","
	      << i.first.first << "," << "FILENAME" << std::endl;
  }
}

//typedef std::list<std::future<std::pair<int,int>>> future_collection_t;
typedef std::list<std::future<QFPTest::resultType>> future_collection_t;
typedef std::chrono::milliseconds const timeout_t;

void checkFutures(future_collection_t& fc, const timeout_t& to,
		  QFPTest::resultType& scores, bool getOne = false){
  for(auto it=fc.begin(); it!=fc.end(); ++it){
    if(it->wait_for(to) != std::future_status::timeout){
      auto val = it->get();
      // for(auto v : val) std::cout << v.first.first << std::endl;
      scores.insert(val.begin(), val.end());
      it = fc.erase(it);
      if(getOne) return;
    }
  }
}

int
main(int argc, char* argv[]){
  char sfx = argv[0][std::strlen(argv[0]) - 1]; //to diff btw inf1 & inf2
  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  std::string TEST;
  loadStringFromEnv(TEST, std::string("TEST") + sfx, "all");
  std::string PRECISION;
  loadStringFromEnv(PRECISION, std::string("PRECISION") + sfx, "all");
  std::string NO_WATCHS;
  loadStringFromEnv(NO_WATCHS, "NO_WATCH", "true");
  bool doOne = TEST != "all";
  if((TEST == "all") != (PRECISION == "all")){ //all or one
    std::cerr << argv[0] << " must be ran with one or all tests selected."
	      << std::endl;
    return 0;
  }  
  int DEGP; //degree of parallelism, or current tasks
  loadIntFromEnv(DEGP, "DEGP", 3);
  std::chrono::milliseconds const timeout (0);
  
  size_t iters = 200;
  size_t dim = 16;
  size_t ulp_inc = 1;
  float min = -6.0;
  float max = 6.0;
  float theta = M_PI;
  
  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

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

    future_collection_t futures;
  
    QFPTest::testInput ip{iters, dim, ulp_inc, min, max};
    scores.clear();
    for(auto& t : TestBase::getTests()){
      auto plist = t.second->create();
      for(auto pt : plist){
	while(DEGP == futures.size()) checkFutures(futures, timeout, scores, false);
	futures.push_back(std::move(std::async(std::launch::async,
					       [pt,ip]{auto retVal =   (*pt)(ip);
						 delete pt; return retVal;})));
	// auto score = (*pt)(ip);
	// scores.insert(score.begin(), score.end());
      }
    }
    while(futures.size() > 0) checkFutures(futures, timeout, scores);
    outputResults(scores);
  }
}


