#ifdef __CUDA__
//#include <cuda.h>
#include "CUHelpers.hpp"
#include "QFPHelpers.hpp"

#define DEVICE 
//#define DEVICE __device__

namespace CUHelpers {

DEVICE
float* cuda_float_rands;  
DEVICE
double* cuda_double_rands;

template<>
DEVICE
float* getRandSeqCU<float>(){return cuda_float_rands;}

template<>
DEVICE
double* getRandSeqCU<double>(){return cuda_double_rands;}

template <typename T>  
//__global__ //TODO fix me
void
loadDeviceData(T** dest, T* source){
  //TODO
  return; //remove me
  *dest = source;
}

void
initDeviceData(){
  return; //TODO remove me, uncomment
  // auto fsize = sizeof(float) * QFPHelpers::float_rands.size();
  // auto dsize = sizeof(double) * QFPHelpers::double_rands.size();
  // float* tfloat;
  // double* tdouble;
  // cudaMalloc(&tfloat,
  //            fsize); 
  // cudaMalloc(&tdouble,
  //            dsize);
  // cudaMemcpy(tfloat,
  //            QFPHelpers::float_rands.data(),
  //            fsize,
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(tdouble,
  //            QFPHelpers::double_rands.data(),
  //            dsize,
  //            cudaMemcpyHostToDevice);
  // loadDeviceData<<<1,1>>>(&cuda_float_rands, tfloat);
  // loadDeviceData<<<1,1>>>(&cuda_double_rands, tdouble);
}

}

#endif
