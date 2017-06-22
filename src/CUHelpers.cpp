#include "CUHelpers.hpp"
#include "flitHelpers.hpp"

namespace flit {

DEVICE float* cuda_float_rands;
DEVICE uint_fast32_t* cuda_16_shuffle;

DEVICE
const float* getRandSeqCU() { return cuda_float_rands; }

DEVICE
const uint_fast32_t* get16ShuffledCU() { return cuda_16_shuffle; }

GLOBAL
void
loadDeviceData(float* fsource, uint_fast32_t* ssource) {
  cuda_float_rands = fsource;
  cuda_16_shuffle = ssource;
}

} // end of namespace flit
