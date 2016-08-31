#ifdef __CUDA__
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "CUHelpers.hpp"
#include "QFPHelpers.hpp"

namespace CUHelpers {

using thrust::device_vector;

__device__
device_vector<float>* cuda_float_rands;
__device__
device_vector<double>* cuda_double_rands;

template <typename T>
__global__
void
loadDeviceData(device_vector<T>** dest, device_vector<T> source){
  *dest = new device_vector<T>(source);
}
  
void
initDeviceData(){
  device_vector<float> frands(QFPHelpers::float_rands.begin(),
                              QFPHelpers::float_rands.end());
  
  device_vector<double> drands(QFPHelpers::double_rands.begin(),
                               QFPHelpers::double_rands.end());
  loadDeviceData<float><<<1,1>>>(&cuda_float_rands,
                                 frands);
  loadDeviceData<double><<<1,1>>>(&cuda_double_rands,
                                  drands);
}

// template<>
// thrust::device_vector<float>
// getRandSeqCU<float>(){
//   return cuda_float_rands;
// }

// template<>
// thrust::device_vector<double>
// getRandSeqCU<double>(){
//   return cuda_double_rands;
// }

}

#endif
