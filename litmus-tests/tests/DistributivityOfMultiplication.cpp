/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END -- */
#include <flit/flit.h>

#include <cmath>
#include <iomanip>
#include <typeinfo>
#include <vector>
#include <tuple>

template <typename T>
class DistributivityOfMultiplication : public flit::TestBase<T> {
public:
  DistributivityOfMultiplication(std::string id)
    : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override;

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T a = ti[0];
    T b = ti[1];
    T c = ti[2];

    auto distributed = (a * c) + (b * c);

    flit::info_stream << std::setw(8);
    flit::info_stream << id << ": (a,b,c) = (" << a << ","
                << b << "," << c << ")" << std::endl;
    flit::info_stream << id << ": dist    = "
                << distributed << std::endl;

    return distributed;
  }

protected:
  using flit::TestBase<T>::id;
};

// Define the inputs
template<>
inline std::vector<float>
DistributivityOfMultiplication<float>::getDefaultInput() {
  auto convert = [](uint32_t x) {
    return flit::as_float(x);
  };

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  std::vector<float> ti = {
    convert(0x6b8b4567),
    convert(0x65ba0c1e),
    convert(0x49e753d2),

    convert(0x233eac52),
    convert(0x22c1532f),
    convert(0x2fda27b0),

    convert(0x2b702392),
    convert(0x3280ef92),
    convert(0x4ece629d),

    convert(0x4f78bee7),
    convert(0x5316ee78),
    convert(0x4f29be1b),

    convert(0x4e27aa59),
    convert(0x4558b7b6),
    convert(0x337f4093),

    convert(0x0e251a94),
    convert(0x060ad983),
    convert(0x702378bd),

    convert(0x3321a89c),
    convert(0x3af748bf),
    convert(0x602dd168),

    convert(0x4e61e16a),
    convert(0x49f3f8fa),
    convert(0x03cc52d0),

    convert(0x5248c931),
    convert(0x5da4cce1),
    convert(0x12384ef7),

    convert(0x58a810f3),
    convert(0x594f3d88),
    convert(0x649f73f0),

    convert(0x07be9118),
    convert(0x00d2636c),
    convert(0x6d984f2b),
  };

  return ti;
}

template<>
inline std::vector<double>
DistributivityOfMultiplication<double>::getDefaultInput() {
  auto convert = [](uint64_t x) {
    return flit::as_float(x);
  };

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  std::vector<double> ti = {
    convert(0x7712d691ff8158c1),
    convert(0x7a71b704fdd6a840),
    convert(0x019b84dddaba0d31),

    convert(0x4443528eec7b4dfb),
    convert(0x43fdcf1f2fb9a656),
    convert(0x7ae870922df32b48),

    convert(0x5b1a5c8d1177cfaa),
    convert(0x595e670dd52ea7bc),
    convert(0x3df3fa0be1c8a9e4),

    convert(0x1be04a5a1d07fe8a),
    convert(0x1a506ba2ab6016be),
    convert(0x57980f57f96de4dc),

    convert(0x4776911a8572ae2e),
    convert(0x47c5c4d1506dcbff),
    convert(0x213ff9f501295930),

    convert(0x29ac0c261d6b14df),
    convert(0x29fc265909b66aab),
    convert(0x69fbe7786470672b),

    convert(0x24b22d74fb8d9e6d),
    convert(0x24c1f1083cc4a7f0),
    convert(0x6c494ff916e4714c),

    convert(0x17d682825d8734bf),
    convert(0x1998785eb236c7ef),
    convert(0x5038e232205d2643),

    convert(0x3774fe15c207a48d),
    convert(0x3a0371c634d95959),
    convert(0x1cfcc1088ead8d5c),

    convert(0x622e8170fa214891),
    convert(0x5f1a608b13e2c398),
    convert(0x4e3491b372540b89),
  };

  return ti;
}

template<>
inline std::vector<long double>
DistributivityOfMultiplication<long double>::getDefaultInput() {
  // Here we are assuming that long double represents 80 bits
  auto convert = [](uint64_t left_half, uint64_t right_half) {
    unsigned __int128 val = left_half;
    val = val << 64;
    val += right_half;
    return flit::as_float(val);
  };

  /// TODO: find values that demonstrate variability with long double
  return {};
}

REGISTER_TYPE(DistributivityOfMultiplication)
