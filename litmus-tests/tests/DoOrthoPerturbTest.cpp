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

#include "Vector.h"

#include <flit.h>

#include <typeinfo>
#include <iomanip>

#include <cmath>

namespace {
  const int iters = 200;
  const int ulp_inc = 1;
}

template <typename T>
class DoOrthoPerturbTest : public flit::TestBase<T> {
public:
  DoOrthoPerturbTest(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 16; }
  virtual std::vector<T> getDefaultInput() override {
    auto dim = getInputsPerRun();
    std::vector<T> ti(dim);
    for(decltype(dim) x = 0; x < dim; ++x)
      ti[x] = static_cast<T>(1 << x);
    return ti;
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    using flit::operator<<;

    auto dim = getInputsPerRun();
    long double score = 0.0;
    std::vector<unsigned> orthoCount(dim, 0.0);
    // we use a double literal above as a workaround for Intel 15-16 compiler
    // bug:
    // https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
    Vector<T> a(ti);
    Vector<T> b = a.genOrthoVector();

    T backup;

    for(decltype(dim) r = 0; r < dim; ++r){
      T &p = a[r];
      backup = p;
      for(int i = 0; i < iters; ++i){
        //cout << "r:" << r << ":i:" << i << std::std::endl;

        p = std::nextafter(p, std::numeric_limits<T>::max());
        // Added this for force watchpoint hits every cycle (well, two).  We
        // shouldn't really be hitting float min
        auto watchPoint = FLT_MIN;
        watchPoint = a ^ b;

        // DELME debug
        //std::cout << watchPoint << std::endl;

        bool isOrth = watchPoint == 0; //a.isOrtho(b);
        if(isOrth){
          orthoCount[r]++;
          // score should be perturbed amount
          if(i != 0) score += std::abs(p - backup);
        }else{
          // if falsely not detecting ortho, should be the dot prod
          if(i == 0) score += std::abs(watchPoint); //a ^ b);
        }
        flit::info_stream
          << "i:" << i
          << ":a[" << r << "] = " << a[r] << ", " << flit::as_int(a[r])
          << " multiplier: " << b[r] << ", " << flit::as_int(b[r])
          << " perp: " << isOrth
          << " dot prod: " << flit::as_int(a ^ b)
          << std::endl;
      }
      flit::info_stream << "next dimension . . . " << std::endl;
      p = backup;
    }
    flit::info_stream << "Final report, one iteration set per dimensiion:" << std::endl;
    flit::info_stream << '\t' << "ulp increment per loop: " << ulp_inc << std::endl;
    flit::info_stream << '\t' << "iterations per dimension: " << iters << std::endl;
    flit::info_stream << '\t' << "dimensions: " << dim << std::endl;
    flit::info_stream << '\t' << "precision (type): " << typeid(T).name() << std::endl;
    int cdim = 0;
    for(auto d: orthoCount){
      int exp = 0;
      std::frexp(a[cdim] * b[cdim], &exp);
      flit::info_stream
        << "For mod dim " << cdim << ", there were " << d
        << " ortho vectors, product magnitude (biased fp exp): " << exp
        << std::endl;
      cdim++;
    }
    return score;
  }

private:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(DoOrthoPerturbTest)
