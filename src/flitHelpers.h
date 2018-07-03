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

// the header for FLiT helper functions.  These classes, such as matrix and
// vector, utilize the TestBase watch data items for monitoring by differential
// debugging.

#ifndef FLIT_HELPERS_HPP
#define FLIT_HELPERS_HPP

#include "InfoStream.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ostream>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include <cfloat>

#ifndef FLIT_UNUSED
#define FLIT_UNUSED(x) (void)x
#endif

namespace flit {

extern thread_local InfoStream info_stream;

std::ostream& operator<<(std::ostream&, const unsigned __int128);
unsigned __int128 stouint128(const std::string &str);

template <typename F, typename I>
F as_float_impl(I val) {
  static_assert(sizeof(F) == sizeof(I), "cannot convert types of different sizes");
  union {
    I i;
    F f;
  } u = { val };
  return u.f;
}

inline float
as_float(uint32_t val) {
  return as_float_impl<float, uint32_t>(val);
}

inline double
as_float(uint64_t val) {
  return as_float_impl<double, uint64_t>(val);
}

inline long double
as_float(unsigned __int128 val) {
  return as_float_impl<long double, __int128>(val);
}

template <typename F, typename I>
I as_int_impl(F val) {
  static_assert(sizeof(F) == sizeof(I), "cannot convert types of different sizes");
  union {
    F f;
    I i;
  } u = { val };
  return u.i;
}

inline uint32_t
as_int(float val) {
  return as_int_impl<float, uint32_t>(val);
}

inline uint64_t
as_int(double val) {
  return as_int_impl<double, uint64_t>(val);
}

inline unsigned __int128
as_int(long double val) {
  const unsigned __int128 zero = 0;
  const auto temp = as_int_impl<long double, __int128>(val);
  return temp & (~zero >> 48);
}

/** Opens a file, but on failure, throws std::ios::failure
 *
 * T must be one of {fstream, ifstream, ofstream}
 *
 * The passed in filestream should be an empty-constructed stream
 */
template <typename T>
void _openfile_check(T& filestream, const std::string &filename) {
  // turn on exceptions on failure
  filestream.exceptions(std::ios::failbit);

  // opening will throw if the file does not exist or is not readable
  filestream.open(filename);

  // turn off all exceptions (back to the default behavior)
  filestream.exceptions(std::ios::goodbit);
}

/** Opens a file for reading, but on failure, throws std::ios::failure
 *
 * The passed in filestream should be an empty-constructed stream
 *
 * Note: This was changed to pass in the stream rather than return it because
 * GCC 4.8 failed to compile - it failed to use the move assignment operator
 * and move constructor, and instead tried to use the copy constructor.
 */
inline void ifopen(std::ifstream& in, const std::string &filename) {
  _openfile_check<std::ifstream>(in, filename);
}

/** Opens a file for writing, but on failure, throws std::ios::failure
 *
 * The passed in filestream should be an empty-constructed stream
 *
 * Note: this was changed to pass in the stream rather than return it because
 * GCC 4.8 failed to compile - it failed to use the move assignment operator
 * and move constructor, and instead tried to use the copy constructor.
 */
inline void ofopen(std::ofstream& out, const std::string &filename) {
  _openfile_check<std::ofstream>(out, filename);
  out.precision(1000);  // lots of digits of precision
}

} // end of namespace flit

#endif // FLIT_HELPERS_HPP

