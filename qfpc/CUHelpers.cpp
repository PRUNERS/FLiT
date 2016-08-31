#ifdef __CUDA__
#include <cuda.h>
#include "CUHelpers.hpp"
#include "QFPHelpers.hpp"

namespace CUHelpers {

__device__
float* cuda_float_rands;  
__device__
double* cuda_double_rands;

template<>
__device__
float* getRandSeqCU<float>(){return cuda_float_rands;}

template<>
__device__
double* getRandSeqCU<double>(){return cuda_double_rands;}

template <typename T>  
__global__
void
loadDeviceData(T** dest, T* source){
  *dest = source;
}

void
initDeviceData(){
  auto fsize = sizeof(float) * QFPHelpers::float_rands.size();
  auto dsize = sizeof(double) * QFPHelpers::double_rands.size();
  float* tfloat;
  double* tdouble;
  cudaMalloc(&tfloat,
             fsize); 
  cudaMalloc(&tdouble,
             dsize);
  cudaMemcpy(tfloat,
             QFPHelpers::float_rands.data(),
             fsize,
             cudaMemcpyHostToDevice);
  cudaMemcpy(tdouble,
             QFPHelpers::double_rands.data(),
             dsize,
             cudaMemcpyHostToDevice);
  loadDeviceData<<<1,1>>>(&cuda_float_rands, tfloat);
  loadDeviceData<<<1,1>>>(&cuda_double_rands, tdouble);
}

}

#endif
