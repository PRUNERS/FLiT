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

template <typename T>
class ReciprocalMath : public flit::TestBase<T> {
public:
  ReciprocalMath(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 5; }

  virtual std::vector<T> getDefaultInput() override {
    return { 7505932213.155222,
	-0.0003670095914968359,
	8780.668938367293,
	-0.0006155790233503232,
	1.0530559711831378,

	8338244879.47946,
	-26.46659826627008,
	-46.167772633949596,
	-8.01886332371321,
	9.389477622336432,

	87.0485837234231,
	814.095091206393,
	376.6986397719769,
	21.457991635073714,
	-34.26603687459364,

	31.594514417427042,
	0.0007624103417064105,
	-0.02577922636404706,
	-0.0009569015524931334,
	-0.0003699749232482463,

	5902221514.636204,
	-0.00021708934100337198,
	9749.762436162175,
	0.8607435476176816,
	-94.28304433351171,

	-49.70966603739113,
	7238595461.899788,
	985.3112995369329,
	-707.7407922965596,
	-671.0617089400928,

	-3699.38938030149,
	-7073528488.719316,
	-164.55786123623116,
	-0.9364155629331614,
	7707.943761459553,

	-62.53791853138796,
	-199.1618358455679,
	-52.036903076794985,
	-2.8204333352680377,
	-2930.340776062591,

	-2852.2836425016662,
	72.33604399822339,
	378.80829294847405,
	9.71642682532883,
	-2.6077212300950436};
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T>& ti) override {
    // float division is expensive.  For optimization, the compiler may replace
    //   a = b / r
    // with
    //   recip = 1 / r
    //   a = b * recip
    // Usually worthwhile only if you divide by the same value repeatedly
    T r = ti[4];
    T a = ti[0] / r;
    T b = ti[1] / r;
    T c = ti[2] / r;
    T d = ti[3] / r;
    return a + b + c + d;
  }
};

REGISTER_TYPE(ReciprocalMath)
