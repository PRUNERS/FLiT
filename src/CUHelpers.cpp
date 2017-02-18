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
  const uint_fast32_t* cuda_16_shuffle = QFPHelpers::shuffled_16.data();
#else
DEVICE
float* cuda_float_rands;
DEVICE
uint_fast32_t* cuda_16_shuffle;
#endif

DEVICE
const float* getRandSeqCU(){return cuda_float_rands;}

DEVICE
const uint_fast32_t* get16ShuffledCU(){return cuda_16_shuffle;}

GLOBAL
void
loadDeviceData(float* fsource, uint_fast32_t* ssource){
  cuda_float_rands = fsource;
  cuda_16_shuffle = ssource;
}

void
initDeviceData(){
#if defined( __CUDA__ ) && !defined( __CPUKERNEL__)
  auto fsize = sizeof(float) * QFPHelpers::float_rands.size();
  auto ssize = sizeof(uint_fast32_t) * QFPHelpers::shuffled_16.size();
  float* tfloat;
  uint_fast32_t* ssource;
  checkCudaErrors(cudaMalloc(&tfloat,
			       fsize)); 
  checkCudaErrors(cudaMemcpy(tfloat,
             QFPHelpers::float_rands.data(),
             fsize,
			     cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&ssource,
			       ssize)); 
  checkCudaErrors(cudaMemcpy(ssource,
             QFPHelpers::shuffled_16.data(),
             ssize,
			     cudaMemcpyHostToDevice));
  loadDeviceData<<<1,1>>>(tfloat, ssource);
  checkCudaErrors(cudaDeviceSynchronize());
#endif
  
}

}

//#endif
