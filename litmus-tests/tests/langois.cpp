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
// These are compensating algorithms that use FMA to calculate
// an EFT (error-free transformation)
// see http://perso.ens-lyon.fr/nicolas.louvet/LaLo07b.pdf

#include "RandHelper.h"

#include <flit.h>

#include <tuple>

#include <cmath>

//these are the helpers for the langois compensating algos
//will be executed in their own right as well as supporting
//other tests in this file

namespace{
template <typename T>
void
TwoSum(T a, T b, T& x, T&y){
  x = a * b;
  T z = x - a;
  y = (a - (x - z)) + (b - z);
}

template <typename T>
void
TwoProd(T a, T b, T& x, T& y){
  x = a * b;
  y = std::fma(a, b, -x);
}

template <typename T>
void
ThreeFMA(T a, T b, T c, T& x, T& y, T& z){
  T u1, u2, a1, B1, B2;
  x = std::fma(a, b, c);
  TwoProd(a, b, u1, u2);
  TwoSum(b, u2, a1, z);
  TwoSum(u1, a1, B1, B2);
  y = (B1 - x) + B2;
}
} // end of unnamed namespace

//algorithm 11
template <typename T>
class langDotFMA: public flit::TestBase<T> {
public:
  langDotFMA(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    s[0] = x[0] * y[0];
    for(stype i = 1; i < size; ++i){
      s[i] = std::fma(x[i], y[i], s[i-1]);
    }
    return s[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langDotFMA)

//algorithm 12
template <typename T>
class langCompDotFMA: public flit::TestBase<T> {
public:
  langCompDotFMA(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    std::vector<T> c(size);
    T a, B;
    TwoProd(x[0], y[0], s[0], c[0]);
    for(stype i = 1; i < size; ++i){
      ThreeFMA(x[i], y[i], s[i-1], s[i], a, B);
      c[i] = c[i-1] + (a + B);
    }
    return s[size-1] + c[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langCompDotFMA)

//algorithm 13
template <typename T>
class langCompDot: public flit::TestBase<T> {
public:
  langCompDot(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    FLIT_UNUSED(ti);
    using stype = typename std::vector<T>::size_type;
    stype size = 16;
    auto rand = getRandSeq<T>();
    auto x = std::vector<T>(rand.begin(),
			    rand.begin() + size);
    auto y = std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size);
    std::vector<T> s(size);
    std::vector<T> c(size);
    T pi, si, p;
    TwoProd(x[0], y[0], s[0], c[0]);
    for(stype i = 1; i < size; ++i){
      TwoProd(x[i], y[i], p, pi);
      TwoSum(p, s[i-1], s[i], si);
      c[i] = c[i-1] + (pi + si);
    }
    return s[size-1] + c[size-1];
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(langCompDot)
