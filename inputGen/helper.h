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

#ifndef HELPER_H
#define HELPER_H

#include <flit/flit.h>

#include <type_traits>
#include <random>
#include <stdexcept>


void printTestVal(const std::string &funcName, float val);
void printTestVal(const std::string &funcName, double val);
void printTestVal(const std::string &funcName, long double val);

uint32_t randGenerator32();
uint64_t randGenerator64();
unsigned __int128 randGenerator128();
float randRealFloatGenerator();
double randRealDoubleGenerator();
long double randRealLongDoubleGenerator();


enum class RandType {
  UniformFP,    // uniform on the space of Floating-Point(FP) numbers
  UniformReals, // uniform on the real number line, then projected to FP
};

template <typename F>
F generateFloatBits(RandType type = RandType::UniformFP);

template<> inline
float generateFloatBits<float>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator32());
    case RandType::UniformReals:
      return randRealFloatGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

template<> inline
double generateFloatBits<double>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator64());
    case RandType::UniformReals:
      return randRealDoubleGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

template<> inline
long double generateFloatBits<long double>(RandType type) {
  switch (type) {
    case RandType::UniformFP:
      return flit::as_float(randGenerator128());
    case RandType::UniformReals:
      return randRealLongDoubleGenerator();
    default:
      throw std::runtime_error("Unimplemented rand type");
  }
}

enum class RandomFloatType {
  Positive,
  Negative,
  Any,
};

template <typename T>
T generateRandomFloat(RandomFloatType fType = RandomFloatType::Any,
                      RandType rType = RandType::UniformFP) {
  static_assert(
      std::is_floating_point<T>::value,
      "generateRandomFloats() can only be used with floating point types"
      );

  // Generate a random floating point number
  T val;
  do {
    val = generateFloatBits<T>(rType);
  } while (std::isnan(val));

  // Convert the values based on the desired qualities
  if (fType == RandomFloatType::Positive) {
    val = std::abs(val);
  } else if (fType == RandomFloatType::Negative) {
    val = -std::abs(val);
  } else if (fType == RandomFloatType::Any) {
    // Do nothing
  } else {
    throw std::runtime_error("Unsupported RandomFloatType passed in");
  }

  return val;
}

#endif // HELPER_H
