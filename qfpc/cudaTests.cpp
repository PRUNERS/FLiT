#include "cudaTests.hpp"
#include "CUHelpers.hpp"    

#ifdef __CUDA__
QFPTest::resultType
DoCudaTest(const QFPTest::testInput& ti,
           std::string id,
           kernelPtr kern,
	   std::string tID,
           uint32_t count = 1){
  cudaResultElement* hresults = new cudaResultElement[count];
#ifdef __CPUKERNEL__
  kern(ti, hresults);
#else
  cudaResultElement* dresults;
  checkCudaErrors(cudaMalloc(&dresults, sizeof(cudaResultElement) * count));
  kern<<<1,1>>>(ti, dresults);
  checkCudaErrors(cudaMemcpy(hresults, dresults, count *
			     sizeof(cudaResultElement),
			     cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(dresults));
#endif
  QFPTest::resultType rt;
  for(uint32_t x = 0; x < count; ++x){
    rt[{id + "_" + std::to_string(x), tID}] =
      {hresults[x].s1, hresults[x].s2};
  }
  delete[] hresults;
  
  return rt;
}
#endif
