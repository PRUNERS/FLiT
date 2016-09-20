#include "QFPHelpers.hpp"
#include "testBase.hpp"
#include "CUHelpers.hpp"
#include <string>



typedef struct {
  double s1;
  double s2;
} cudaResultElement;

//typedef void (*kernelPtr)(cudaResultElement*);
typedef void (*kernelPtr)(const QFPTest::testInput, cudaResultElement*);

QFPTest::resultType
DoCudaTest(const QFPTest::testInput&,
           std::string id,
           kernelPtr&,
	   std::string tID,
	   uint32_t);
