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
#include "RandHelper.h"

#include <flit.h>

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <type_traits>

template <typename T>
class FtoDecToF: public flit::TestBase<T> {
public:
  FtoDecToF(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { std::numeric_limits<T>::min() };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    std::numeric_limits<T> nlim;
    // from https://en.wikipedia.org/wiki/IEEE_floating_point
    uint16_t ddigs = nlim.digits * std::log10(2) + 1;
    std::ostringstream res;
    res << std::setprecision(ddigs) << ti[0];
    std::string dstr;
    dstr = res.str();
    T backAgain;
    std::istringstream(dstr) >> backAgain;
    return ti[0] - backAgain;
  }

  using flit::TestBase<T>::id;
};
REGISTER_TYPE(FtoDecToF)

template <typename T>
class subnormal: public flit::TestBase<T> {
public:
  subnormal(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { std::numeric_limits<T>::min() };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    return ti[0] - ti[0] / 2;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(subnormal)

template <typename T>
class dotProd: public flit::TestBase<T> {
public:
  dotProd(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    FLIT_UNUSED(ti);
    auto size = 16;

    auto rand = getRandSeq<T>();

    Vector<T> A(std::vector<T>(rand.begin(),
			    rand.begin() + size));
    Vector<T> B(std::vector<T>(rand.begin() + size,
			    rand.begin() + 2*size));
    return A ^ B;
  }

protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(dotProd)

template <typename T>
class simpleReduction: public flit::TestBase<T> {
public:
  simpleReduction(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    FLIT_UNUSED(ti);
    auto vals = getRandSeq<T>();
    auto sublen = vals.size() / 4 - 1;
    T sum = 0;
    for(uint32_t i = 0; i < sublen; i += 4){
      sum += vals[i];
      sum += vals[i+1];
      sum += vals[i+2];
      sum += vals[i+3];
    }
    for(uint32_t i = sublen; i < vals.size(); ++i){
      sum += vals[i];
    }
    return sum;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(simpleReduction)

template <typename T>
class addTOL : public flit::TestBase<T> {
public:
  addTOL(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(man_bits + 1, nls.max_exponent);
    // generate the range of offsets, and then generate the
    // mantissa bits for each of the three inputs
    auto L1e = dis(gen); //L1 exponent

    // for the ldexp function we're using, it takes an unbiased exponent and
    // there is no implied 1 MSB for the mantissa / significand
    T zero = 0.0;
    auto L1m = flit::as_int(zero);
    auto L2m = flit::as_int(zero);
    auto sm = flit::as_int(zero);
    for(int i = 0; i < man_bits; ++i){
      L1m &= (gen() & 1) << i;
      L2m &= (gen() & 1) << i;
      sm  &= (gen() & 1) << i;
    }
    return {
      std::ldexp(flit::as_float(L1m), L1e),
      std::ldexp(flit::as_float(L2m), L1e - 1),
      std::ldexp(flit::as_float(sm), L1e - man_bits)
    };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] + ti[1] + ti[2];
    return res;
  }

  using flit::TestBase<T>::id;
};
REGISTER_TYPE(addTOL)

template <typename T>
class addSub: public flit::TestBase<T> {
public:
  addSub(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { T(1.0) };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    std::numeric_limits<T> nls;
    auto man_bits = nls.digits;
    auto big = std::pow(2, (T)man_bits - 1);
    auto res = (ti[0] + big) - big;
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(addSub)

template <typename T>
class divc: public flit::TestBase<T> {
public:
  divc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
    };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] / ti[1];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(divc)

template <typename T>
class zeroMinusX: public flit::TestBase<T> {
public:
  zeroMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = T(0.0) - ti[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroMinusX)

template <typename T>
class xMinusZero: public flit::TestBase<T> {
public:
  xMinusZero(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] - T(0.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusZero)

template <typename T>
class zeroDivX: public flit::TestBase<T> {
public:
  zeroDivX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = (T)0.0 / ti[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(zeroDivX)

template <typename T>
class xDivOne: public flit::TestBase<T> {
public:
  xDivOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T res = ti[0] / T(1.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivOne)

template <typename T>
class xDivNegOne: public flit::TestBase<T> {
public:
  xDivNegOne(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    T res = ti[0] / T(-1.0);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xDivNegOne)

template <typename T>
class negAdivB: public flit::TestBase<T> {
public:
  negAdivB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] / ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAdivB)

template <typename T>
class negAminB: public flit::TestBase<T> {
public:
  negAminB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] - ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAminB)

template <typename T>
class xMinusX: public flit::TestBase<T> {
public:
  xMinusX(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { getRandSeq<T>()[0] };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] - ti[0];
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xMinusX)

template <typename T>
class negAplusB: public flit::TestBase<T> {
public:
  negAplusB(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 2; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = -(ti[0] + ti[1]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(negAplusB)

template <typename T>
class aXbDivC: public flit::TestBase<T> {
public:
  aXbDivC(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
      getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] * (ti[1] / ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbDivC)

template <typename T>
class aXbXc: public flit::TestBase<T> {
public:
  aXbXc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
      getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] * (ti[1] * ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aXbXc)

template <typename T>
class aPbPc: public flit::TestBase<T> {
public:
  aPbPc(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    return {
      getRandSeq<T>()[0],
      getRandSeq<T>()[1],
      getRandSeq<T>()[2],
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    auto res = ti[0] + (ti[1] + ti[2]);
    return res;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(aPbPc)

template <typename T>
class xPc1EqC2: public flit::TestBase<T> {
public:
  xPc1EqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    const T eps = std::numeric_limits<T>::min();
    const T next = std::nextafter(eps, std::numeric_limits<T>::infinity());
    return {
      getRandSeq<T>()[0],
      eps,
      next,
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    bool res = ti[0] + ti[1] == ti[2];
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1EqC2)

template <typename T>
class xPc1NeqC2: public flit::TestBase<T> {
public:
  xPc1NeqC2(std::string id) : flit::TestBase<T>(std::move(id)){}

  virtual size_t getInputsPerRun() override { return 3; }
  virtual std::vector<T> getDefaultInput() override {
    const T eps = std::numeric_limits<T>::min();
    const T next = std::nextafter(eps, std::numeric_limits<T>::infinity());
    return {
      getRandSeq<T>()[0],
      eps,
      next,
    };
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    bool res = (ti[0] + ti[1] != ti[2]);
    return res ? 1.0 : 0.0;
  }
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(xPc1NeqC2)
