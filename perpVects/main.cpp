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
typedef std::map<std::pair<std::string, std::string>,
                 std::pair<long double, long double>> score_t;

typename QFPHelpers::sort_t
getSortT(int t){
  switch(t){
  case 0:
    return QFPHelpers::lt;
  case 1:
    return QFPHelpers::gt;
  case 2:
    return QFPHelpers::bi;
  case 3:
    return QFPHelpers::def;
  default:
    return QFPHelpers::def;
  }
}

int
getSortID(std::string s){
  if(s == "lt") return lt;
  if(s == "gt") return gt;
  if(s == "bi") return bi;
  return def;
}

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
outputResults(const QFPTest::testInput& ti, const score_t& scores){
  for(const auto& i: scores){
    std::cout << "HOST,SWITCHES,COMPILER," << i.first.second << "," << getSortName(ti.reduction_sort_type)
         << "," << i.second.first << "," << FPWrap<long double>(i.second.first) << "," <<
      i.second.second << "," << FPWrap<long double>(i.second.second) << "," << 
      i.first.first << "," << "FILENAME" << std::endl;
  }
}

//typedef std::list<std::future<std::pair<int,int>>> future_collection_t;
typedef std::list<std::future<QFPTest::resultType>> future_collection_t;
typedef std::chrono::milliseconds const timeout_t;

void checkFutures(future_collection_t& fc, const timeout_t& to, score_t& scores){
  for(auto it=fc.begin(); it!=fc.end(); ++it){
    // std::cout << "waiting on future . . ." << std::endl;
    if(it->wait_for(to) != std::future_status::timeout){
      // std::cout << "finished wait, no timeout" << std::endl;
      // std::cout << "future valid: " << it->valid() << std::endl;
      auto val = it->get();
      // std::cout << val << std::endl;
      scores.insert(val);
      // std::cout << "fetched result (get())" << std::endl;
      it = fc.erase(it);
    }
  }
}

int
main(int argc, char* argv[]){
  char sfx = argv[0][std::strlen(argv[0]) - 1]; //to diff btw inf1 & inf2
  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  std::string TEST;
  loadStringFromEnv(TEST, std::string("TEST") + sfx, "all");
  std::string SORT;
  loadStringFromEnv(SORT, std::string("SORT") + sfx, "all");
  std::string PRECISION;
  loadStringFromEnv(PRECISION, std::string("PRECISION") + sfx, "all");
  std::string NO_WATCHS;
  loadStringFromEnv(NO_WATCHS, "NO_WATCH", "true");
  bool doOne = TEST != "all";
  std::cout << TEST << ":" << SORT << ":" << PRECISION << std::endl;
  if(TEST == "all"){
    if(SORT != "all" || PRECISION != "all"){ //all or one
    std::cerr << argv[0] << " must be ran with one or all tests selected." << std::endl;
    return 0;
    }
  }else{
    if(SORT == "all" || PRECISION == "all"){
      std::cerr << argv[0] << " must be ran with one or all tests selected." << std::endl;
      return 0;
    }
  }
  
  int DEGP; //degree of parallelism, or current tasks
  loadIntFromEnv(DEGP, "DEGP", 8);
  std::chrono::milliseconds const timeout (100);
  
  size_t iters = 200;
  size_t dim = 16;
  size_t ulp_inc = 1;
  float min = -6.0;
  float max = 6.0;
  float theta = M_PI;
  
  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

  score_t scores;

  scores.clear();

  if(NO_WATCHS != "true"){
    QFPTest::setWatching();
  }

  //singleton
  if(doOne){
    QFPTest::testInput ip{iters, dim, ulp_inc, min, max,
        getSortT(getSortID(SORT))};
    auto plist = TestBase::getTests()[TEST]->create();
    auto score = (*plist[getPrecID(PRECISION)])(ip);
    for(auto& p: plist) delete p;
    scores.insert(score);
    outputResults(ip, scores);
  }else{

    future_collection_t futures;
  
    for(int ipm = 0; ipm < 4; ++ipm){ //reduction sort pre sum
      //std::cout << "starting test set on precision: " << ipm << std::endl;
      QFPTest::testInput ip{iters, dim, ulp_inc, min, max,
	  getSortT(ipm)};
      scores.clear();
      for(auto& t : TestBase::getTests()){
	auto plist = t.second->create();
	while(DEGP - futures.size() < plist.size()) checkFutures(futures, timeout, scores);
	for(auto pt : plist){
	  //futures.push_back(std::move(std::async(&QFPTest::TestBase::operator(), pt, ip)));
	  futures.push_back(std::move(std::async(std::launch::async, [pt,ip]{auto retVal =   (*pt)(ip);
		  delete pt; return retVal;})));
	  //futures.push_back(std::move(std::async([pt,ip]{return (QFPTest::resultType){{std::string("hi"), std::string("there")},{0.0,0.0}};})));
	  // scores.insert((*pt)(ip));
	  // scores.insert([pt,ip]{return (*pt)(ip);}());
	}
	//for(auto t : plist) delete t;
      }
      while(futures.size() > 0) checkFutures(futures, timeout, scores);
      outputResults(ip, scores);
    }
  }
}


