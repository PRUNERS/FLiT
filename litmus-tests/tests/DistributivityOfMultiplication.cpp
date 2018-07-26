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
#include <flit.h>

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

  // Put in canned values of previously found diverging inputs
  // These are entered as hex values to maintain the exact value instead of trying
  // to specify enough decimal digits to get the same floating-point value
  std::vector<long double> ti = {
      convert(0x2b99, 0x2bb4d082ca2e7ec7),  //  3.586714e-1573
      convert(0x725a, 0x14c0a0cd445b52d5),  //  6.131032e+3879
      convert(0x075d, 0x0bc91b713fc2fba5),  //  4.278225e-4366

      convert(0x3408, 0xd98776d83be541b8),  //  1.497721e-922
      convert(0x7da5, 0x32daa5df77e78b5e),  //  2.847787e+4750
      convert(0x376a, 0xfa52e8946985dab4),  //  8.479921e-662

      convert(0x2355, 0xb32ca57fbcc68a6c),  //  1.541551e-2209
      convert(0x7337, 0x4855e1d4f174504d),  //  7.201858e+3946
      convert(0x736a, 0xac4f338d852e88cd),  //  3.863064e+3962

      convert(0x4727, 0x9c8f934e1cc682d2),  //  3.753403e+551
      convert(0x02e7, 0x3b753c2d81c6bf78),  //  3.612998e-4709
      convert(0x485b, 0xeae81af41947d10a),  //  2.936812e+644

      convert(0x1b91, 0x9e18ddbb66670e9f),  //  4.852600e-2808
      convert(0x205f, 0x58ab17d3e5309234),  //  5.031688e-2438
      convert(0x4a4a, 0x1b8a0c6541700676),  //  3.521930e+792

      convert(0x4178, 0xd2cee20bba5c6843),  //  5.069741e+113
      convert(0x3564, 0x58406d82dd970b8e),  //  3.483973e-818
      convert(0x150a, 0x92cde10402bc42ef),  //  4.292064e-3311

      convert(0x1965, 0x630847ac1ecd253c),  //  1.288694e-2975
      convert(0x004c, 0x6e2b4c6a070d3835),  //  1.093228e-4909
      convert(0x380f, 0x92b14ca6d81b5a24),  //  2.324058e-612

      convert(0x4492, 0x870e870425dcb0cf),  //  3.384007e+352
      convert(0x71dd, 0x159330946cecd9a8),  //  1.498527e+3842
      convert(0x586a, 0xfc38e15fe5d604a5),  //  1.079136e+1882

      convert(0x240d, 0xae73609d2bf51b7d),  //  3.680220e-2154
      convert(0x2a67, 0x89b93255d3362c94),  //  8.669256e-1665
      convert(0x2462, 0x79d020dd3c308e90),  //  9.941326e-2129

      convert(0x6703, 0x6455d50eb2825cf7),  //  3.818039e+3006
      convert(0x1f77, 0x70b75c7169817349),  //  9.267715e-2508
      convert(0x4ab4, 0x27faa40914dad6a6),  //  4.148019e+824
  };

  return ti;
}

REGISTER_TYPE(DistributivityOfMultiplication)
