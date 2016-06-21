// This is the main implementation, handling cmd line params and
// running the tests!

#include <cstring>
#include <typeinfo>
#include <tuple>

#include "testBase.h"
#include "QFPHelpers.h"


using namespace QFPHelpers;
using namespace QFPTest;

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
  std::cout << "in lsfe, var is: " << var << std::endl;
  if(std::getenv(var.c_str()) == NULL){
    dest = defVal;
  }else{
    dest = std::getenv(var.c_str());
  }
  std::cout << "env is: " << dest << std::endl;
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

void
doTestSet(std::vector<TestBase*> pSet,
	  QFPTest::testInput &ip,
	  score_t& scores,
	  int prec = -1){
  if(prec == -1){
    
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
  std::cout << sfx << ":" << TEST << "," << SORT << "," <<
    PRECISION << "," << NO_WATCHS << std::endl;
  //TODO really, let's clean up.  We don't need to worry about
  // defaults.  Either we're running a single test, with
  // watch enabled, or we're running them all, I think.
  // Unless we want to rerun a single test without watch
  // (and update DB).

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

  int firstST;
  int lastST;
  if(NO_WATCHS != "true"){
    QFPTest::setWatching();
  }
  if(SORT != "all") {
    firstST = getSortID(SORT);
    lastST = getSortID(SORT) + 1;
  }else{
    firstST = 0;
    lastST = 4;
  }

  // std::cout << "size of getTests return: " << TestBase::getTests().size() << std::endl;
  auto &testSet = TestBase::getTests();
  if(TEST != "all"){
    testSet = {{TEST, TestBase::getTests()[TEST]}};
  }
  
  for(int ipm = firstST; ipm < lastST; ++ipm){ //reduction sort pre sum
    //std::cout << "starting test set on precision: " << ipm << std::endl;
    QFPTest::testInput ip{iters, dim, ulp_inc, min, max,
	getSortT(ipm)};
    for(auto& t : testSet){
      scores.clear();
      auto plist = t.second->create();
      // std::cout << "running precision set on " << t.first << std::endl;
      // std::cout << "size of plist is: " << plist.size() << std::endl;
      if(PRECISION != "all"){
	auto score = (*plist[getPrecID(PRECISION)])(ip);
	scores.insert(score);
	// std::cout << score << std::endl;
      }else{
	for(auto pt : plist){
	  auto score = (*pt)(ip);
	  scores.insert(score);
	  // std::cout << score << std::endl;
	}
      }
      for(auto t : plist) delete t;
      outputResults(ip, scores);
    }
  }
}


