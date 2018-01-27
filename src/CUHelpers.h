#ifndef CU_HELPERS_HPP
#define CU_HELPERS_HPP

#if defined(__CPUKERNEL__) || !defined( __CUDA__)
#define HOST_DEVICE
#define HOST
#define DEVICE
#define GLOBAL
#else
#include <cuda.h>
#include <helper_cuda.h>
#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#endif
#include "flitHelpers.h"
#include "cuvector.h"

#include <vector>

#include <cmath>

namespace flit {

// TODO: test out trying to replace csqrt() with std::sqrt()
template <typename T>
T
csqrt(T /*val*/){ return 0; }

template<>
HOST_DEVICE
inline
float
csqrt<float>(float val) { return sqrtf(val); }

template<>
HOST_DEVICE
inline
double
csqrt<double>(double val) { return sqrt(val);}

template <typename T>
T
cpow(T /*a*/, T /*b*/){ return 0; }

template<>
HOST_DEVICE
inline
float
cpow<float>(float a, float b) { return powf(a,b); }

template<>
HOST_DEVICE
inline
double
cpow<double>(double a, double b) { return pow(a,b); }

template <typename T>
T
ccos(T /*val*/){ return 0; }

template<>
HOST_DEVICE
inline
float
ccos<float>(float val){ return cosf(val); }


template<>
HOST_DEVICE
inline
double
ccos<double>(double val){ return cos(val); }

template <typename T>
T
csin(T /*val*/){ return 0; }

template<>
HOST_DEVICE
inline
float
csin<float>(float val){ return sinf(val); }

template<>
HOST_DEVICE
inline
double
csin<double>(double val){ return sin(val); }

void
initDeviceData();

DEVICE
const float* getRandSeqCU();

DEVICE
const uint_fast32_t* get16ShuffledCU(); //an array with 0-15 shuffled

template <typename T>
HOST_DEVICE
T
abs(T val){
  if(val > 0) return val;
  else return val * (T)-1;
}

extern const std::vector<float> float_rands;
extern const std::vector<double> double_rands;
extern const std::vector<long double> long_rands;
extern const std::vector<uint_fast32_t> shuffled_16;

GLOBAL void loadDeviceData(float* fsource, uint_fast32_t* ssource);


inline void
initDeviceData() {
#ifdef __CPUKERNEL__
  cuda_float_rands = flit::float_rands.data();
  cuda_16_shuffle = flit::shuffled_16.data();
#endif // __CPUKERNEL__
#if defined(__CUDA__) && !defined(__CPUKERNEL__)
  auto fsize = sizeof(float) * flit::float_rands.size();
  auto ssize = sizeof(uint_fast32_t) * flit::shuffled_16.size();
  float* tfloat;
  uint_fast32_t* ssource;
  checkCudaErrors(cudaMalloc(&tfloat,
             fsize));
  checkCudaErrors(cudaMemcpy(tfloat,
             flit::float_rands.data(),
             fsize,
             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&ssource,
             ssize));
  checkCudaErrors(cudaMemcpy(ssource,
             flit::shuffled_16.data(),
             ssize,
             cudaMemcpyHostToDevice));
  loadDeviceData<<<1,1>>>(tfloat, ssource);
  checkCudaErrors(cudaDeviceSynchronize());
#endif // defined(__CUDA__) && !defined(__CPUKERNEL__)
}

} // end of namespace flit

#endif // CU_HELPERS_HPP
