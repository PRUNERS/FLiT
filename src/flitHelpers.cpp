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
 * -- LICENSE END --
 */

// this contains the helper classes for the tests.
// they utilize the watch data for sensitive points
// of computation.

#include "flitHelpers.h"

#include <iostream>
#include <mutex>

namespace flit {

thread_local InfoStream info_stream;

std::ostream& operator<<(std::ostream& os, const unsigned __int128 i){
  std::ostringstream ost;
  uint64_t hi = i >> 64;
  uint64_t lo = (uint64_t)i;
  auto bflags = os.flags();
  os.flags(std::ios::hex & ~std::ios::showbase);
  ost.flags(std::ios::hex & ~std::ios::showbase);
  ost << lo;
  os << "0x" << hi;
  for(uint32_t x = 0; x < 16 - ost.str().length(); ++x){
    os << "0";
  }
  os << ost.str();
  os.flags( bflags );
  return os;
}

unsigned __int128 stouint128(const std::string &str) {
  uint64_t hi, lo;
  // TODO: make this more efficient (maybe).
  std::string copy;
  if (str[0] == '0' && str[1] == 'x') {
    copy = std::string(str.begin() + 2, str.end());
  } else {
    copy = str;
  }

  // Convert each section of 8-bytes (16 characters)
  if (copy.size() > 32) {
    throw std::invalid_argument("Too many digits to convert with stouint128");
  }
  if (copy.size() <= 16) {
    hi = 0;
    lo = std::stoull(copy, nullptr, 16);
  } else {
    auto mid = copy.end() - 16;
    hi = std::stoull(std::string(copy.begin(), mid), nullptr, 16);
    lo = std::stoull(std::string(mid, copy.end()), nullptr, 16);
  }

  // Combine the two 64-bit values.
  unsigned __int128 val;
  val = hi;
  val = val << 64;
  val += lo;
  return val;
}

} // end of namespace flit

