//#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.hpp"
#include "QFPHelpers.hpp"
#ifdef __CUDA__
#include <helper_cuda.h>
#endif

namespace CUHelpers {

#ifdef __CPUKERNEL__
  const float* cuda_float_rands = QFPHelpers::float_rands.data();
#else
DEVICE
float* cuda_float_rands;
#endif
// DEVICE
// double* cuda_double_rands;


HOST_DEVICE
const float* getRandSeqCU(){return cuda_float_rands;}
// template<>
// DEVICE
// double* getRandSeqCU<double>(){return cuda_double_rands;}

GLOBAL
void
loadDeviceData(float* fsource){
  cuda_float_rands = fsource;
  // for(uint32_t x = 0; x < 32; ++x){
  //   printf("%f\n", cuda_float_rands[x]);
  // }
}

void
initDeviceData(){
#if defined( __CUDA__ ) && !defined( __CPUKERNEL__)
  auto fsize = sizeof(float) * QFPHelpers::float_rands.size();
  float* tfloat;
  checkCudaErrors( cudaMalloc(&tfloat,
			       fsize)); 
  checkCudaErrors(cudaMemcpy(tfloat,
             QFPHelpers::float_rands.data(),
             fsize,
			     cudaMemcpyHostToDevice));
  loadDeviceData<<<1,1>>>(tfloat);
  checkCudaErrors(cudaDeviceSynchronize());
#endif
  
}

}

//#endif
