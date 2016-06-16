// This is the main implementation, handling cmd line params and
// running the tests!

#include <cstring>
#include <typeinfo>

#include "testBase.h"
#include "QFPHelpers.h"


using namespace QFPHelpers;
using namespace QFPTest;

typedef std::map<std::string,
		 std::pair<long double, long double>> score_t;

template<typename T>
void
DoTests(const testInput& ti, score_t &scores){
  if(typeid(float) == typeid(T)){
    for(auto &tfact: TestBase::getTests()){
      scores.insert(tfact.second->create()->floatTest(ti));
    }
    return;
  }
  if(typeid(double) == typeid(T)){
    for(auto &tfact: TestBase::getTests()){
      scores.insert(tfact.second->create()->doubleTest(ti));
    }
    return;
  }
  if(typeid(long double) == typeid(T)){
    for(auto &tfact: TestBase::getTests()){
      scores.insert(tfact.second->create()->longTest(ti));
    }
    return;
  }
  throw std::string("unknown type instantiation of ") +
    std::string(__func__);
}


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
  if(std::getenv(var.c_str()) == NULL){
    dest = defVal;
  }else{
    dest = std::getenv(var.c_str());
  }
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

template<typename T>
void
outputResults(const QFPTest::testInput& ti, score_t &scores){
  for(auto i: scores){
    std::cout << "HOST,SWITCHES,COMPILER," << typeid(T).name() << "," << getSortName(ti.reduction_sort_type)
	 << "," << i.second.first << "," << FPWrap<long double>(i.second.first) << "," <<
      i.second.second << "," << FPWrap<long double>(i.second.second) << "," << 
      i.first << "," << "FILENAME" << std::endl;
  }
}

int
main(int argc, char* argv[]){
  char sfx = argv[0][std::strlen(argv[0]) - 1]; //to diff btw inf1 & inf2
  if(argc > 1 && std::string(argv[1]) == std::string("verbose")) info_stream.show();
  std::string TEST;
  loadStringFromEnv(TEST, "TEST", "all");
  std::string SORT;
  loadStringFromEnv(SORT, std::string("SORT") + sfx, "all");
  std::string PRECISION;
  loadStringFromEnv(PRECISION, std::string("PRECISION") + sfx, "all");
  std::string NO_WATCHS;
  loadStringFromEnv(NO_WATCHS, "NO_WATCH", "true");
  //THIS EXTERNAL INTERFACE IS A LITTLE SLOPPY -- SHOULD REENGINEER

  //  using namespace fpTestSuite;
  //using namespace QFP;
  // The params to perturb are:
  size_t iters = 200;
  size_t dim = 16;
  size_t ulp_inc = 1;
  float min = -6.0;
  float max = 6.0;
  // theta [for rotation tests]
  float theta = M_PI;
  
  std::cout.precision(1000); //set cout to print many decimal places
  info_stream.precision(1000);

  score_t scores;

  int firstST = 0;
  int lastST = 4;
  int firstP = 0;
  int lastP = 3;
  if(SORT != "all"){
    firstST = getSortID(SORT);
    lastST = getSortID(SORT) + 1;
  }
  if(PRECISION != "all"){
    firstP = getPrecID(PRECISION);
    lastP = getPrecID(PRECISION) + 1;
  }
  if(NO_WATCHS != "true"){
    QFPTest::setWatching();
  }
  //TODO we need to add the next conditional to someplace in the loop;
  //I suppose that this means we let the loop bounds guide how many
  //further types we run
  for(int ipm = firstST; ipm < lastST; ++ipm){ //reduction sort pre sum
    for(int p = firstP; p < lastP; ++p){ //float, double, long double
      scores.clear();
      switch(p){
      case 0: //float
	{
	  QFPTest:: testInput ip{iters, dim, ulp_inc, min, max,
	      getSortT(ipm)};	  
	  if (TEST != "all"){
	    scores.insert(TestBase::getTests()[TEST]->create()->floatTest(ip));
	  }else{
	    DoTests<float>(ip, scores);
	  }
	  outputResults<float>(ip, scores);
	  break;
	}
      case 1:
	{
	  QFPTest:: testInput ip{iters, dim, ulp_inc, min, max,
	      getSortT(ipm)};
	  if (TEST != "all"){
	    scores.insert(TestBase::getTests()[TEST]->create()->doubleTest(ip));
	  }else{
	    DoTests<double>(ip, scores);
	  }
	  outputResults<double>(ip, scores);
	  break;
	}
      case 2:
	{
	  QFPTest:: testInput ip{iters, dim, ulp_inc, min, max,
	      getSortT(ipm)};
	  if (TEST != "all"){
	    scores.insert(TestBase::getTests()[TEST]->create()->longTest(ip));
	  }else{
	    DoTests<long double>(ip, scores);
	  }
	  outputResults<long double>(ip, scores);
	  break;
	}
      }
    }
  }
}

