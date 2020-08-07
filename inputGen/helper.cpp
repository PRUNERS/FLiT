/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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
 *   https://github.com/PRUNERS/FLiT/blob/main/LICENSE
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
 * -- LICENSE END --
 */

#include "helper.h"

#include <flit/flit.h>

#include <iostream>
#include <iomanip>

/// RAII class for restoring iostream formats
class FmtRestore {
public:
  FmtRestore(std::ios& stream) : _stream(stream), _state(nullptr) {
    _state.copyfmt(_stream);
  }
  ~FmtRestore() { _stream.copyfmt(_state); }
private:
  std::ios& _stream;
  std::ios  _state;
};

void printTestVal(const std::string &funcName, float val) {
  FmtRestore restorer(std::cout);
  FLIT_UNUSED(restorer);

  auto intval = flit::as_int(val);
  std::cout << funcName << ":     0x"
            << std::hex << std::setw(8) << std::setfill('0') << intval
            << "  "
            << std::scientific << val
            << std::endl;
}

void printTestVal(const std::string &funcName, double val) {
  FmtRestore restorer(std::cout);
  FLIT_UNUSED(restorer);

  auto intval = flit::as_int(val);
  std::cout << funcName << ":     0x"
            << std::hex << std::setw(16) << std::setfill('0') << intval
            << "  "
            << std::scientific << val
            << std::endl;
}

void printTestVal(const std::string &funcName, long double val) {
  FmtRestore restorer(std::cout);
  FLIT_UNUSED(restorer);

  auto intval = flit::as_int(val);
  uint64_t lhalf = static_cast<uint64_t>((intval >> 64)) & 0xFFFFL;
  uint64_t rhalf = static_cast<uint64_t>(intval);

  std::cout << funcName << ":     0x"
            << std::hex << std::setw(4) << std::setfill('0') << lhalf
            << std::hex << std::setw(16) << std::setfill('0') << rhalf
            << "  "
            << std::scientific << val
            << std::endl;
}

namespace {

  auto generateSeed() {
    std::random_device seedGenerator;
    return seedGenerator();
  }

} // end of unnamed namespace

uint32_t randGenerator32() {
  static auto seed = generateSeed();
  static std::mt19937 generator32(seed);
  return generator32();
}

uint64_t randGenerator64() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  return generator64();
}

unsigned __int128 randGenerator128() {
  static auto generator128 = []() {
    unsigned __int128 val = randGenerator64();
    val = val << 64;
    val += randGenerator64();
    return val;
  };
  return generator128();
}

float randRealFloatGenerator() {
  static auto seed = generateSeed();
  static std::mt19937 generator32(seed);
  static std::uniform_real_distribution<float> distribution(
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max()
      );
  return distribution(generator32);
}

double randRealDoubleGenerator() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  static std::uniform_real_distribution<double> distribution(
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max()
      );
  return distribution(generator64);
}

long double randRealLongDoubleGenerator() {
  static auto seed = generateSeed();
  static std::mt19937_64 generator64(seed);
  static std::uniform_real_distribution<long double> distribution(
      std::numeric_limits<long double>::min(),
      std::numeric_limits<long double>::max()
      );
  return distribution(generator64);
}
