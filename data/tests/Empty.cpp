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

#include <string>

/** An example test class to show how to make FLiT tests
 *
 * You will want to rename this file and rename the class.  Then implement
 * getInputsPerRun(), getDefaultInput() and run_impl().
 */
template <typename T>
class Empty : public flit::TestBase<T> {
public:
  Empty(std::string id) : flit::TestBase<T>(std::move(id)) {}

  /** Specify how many floating-point inputs your algorithm takes.
   *
   * Can be zero.  If it is zero, then getDefaultInput should return an empty
   * std::vector, which is as simple as "return {};"
   */
  virtual size_t getInputsPerRun() override { return 1; }

  /** Specify the default inputs for your test.
   *
   * Used for automated runs.  If you give number in ti.vals than
   * getInputsPerRun, then the run_impl will get called more than once, each
   * time with getInputsPerRun() elements in ti.vals.
   *
   * If your algorithm takes no inputs, then you can simply return an empty
   * std::vector object.  It is as simple as "return {};".
   */
  virtual std::vector<T> getDefaultInput() override {
    return { 1.0 };
  }

  /** Custom comparison methods
   *
   * These comparison operations are meant to create a metric between the test
   * results from this test in the current compilation, and the results from
   * the ground truth compilation.  You can do things like the relative error
   * or the absolute error (for the case of long double).
   *
   * The below specified functions are the default implementations defined in
   * the base class.  It is safe to delete these two functions if this
   * implementation is adequate for you.
   *
   * Which one is used depends on the type of Variant that is returned from the
   * run_impl function.  The value returned by compare will be the value stored
   * in the database for later analysis.
   */
  virtual long double compare(long double ground_truth,
                              long double test_results) const override {
    // absolute error
    return test_results - ground_truth;
  }

  /** There is no good default implementation comparing two strings */
  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    FLIT_UNUSED(ground_truth);
    FLIT_UNUSED(test_results);
    return 0.0;
  }

protected:
  /** Call or implement the algorithm here.
   *
   * You return a flit::Variant which can represent one of a number of
   * different types.  See the flit::Variant class for more details.  In short,
   * it can represent a long double, a std::string, or an empty nothing (at
   * least as of this writing.  See the flit::Variant documentation for more
   * up-to-date information).  If you return an empty flit::Variant, then that
   * result will not show up in the output (meaning that test is disabled).
   * This is desirable for example if you only have functionality with double
   * precision and want to disable the float and long double test
   * implementations (simply have those return an empty flit::Variant).
   *
   * The value returned by run_impl is the same value used in compare()
   * implemented above.
   */
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(Empty)
