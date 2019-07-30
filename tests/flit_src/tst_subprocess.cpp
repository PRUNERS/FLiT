/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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

// We need to add some stuff to main() to get call_main() to work
#define main th_main
#include "test_harness.h"
#undef main

#include "subprocess.h"
#include "flitHelpers.h"
#include "flit.h"
#include "fsutil.h"

int main(int argCount, char* argList[]) {
  // Fast track means calling a user's main() function
  if (flit::isFastTrack(argCount, argList)) {
    return flit::callFastTrack(argCount, argList);
  }
  // otherwise, run tests
  return th_main(argCount, argList);
}

void tst_subprocess() {
  using flit::operator<<;

  auto retval = flit::call_with_output("echo this output is expected");
  TH_EQUAL(0, retval.ret);
  TH_EQUAL("this output is expected\n", retval.out);
  TH_EQUAL("", retval.err);

  retval = flit::call_with_output("false");
  TH_EQUAL(1, retval.ret);
  TH_EQUAL("", retval.out);
  TH_EQUAL("", retval.err);

  retval = flit::call_with_output("echo hello | grep nope");
  TH_EQUAL(1, retval.ret);
  TH_EQUAL("", retval.out);
  TH_EQUAL("", retval.err);

  retval = flit::call_with_output("echo hello | grep hello");
  TH_EQUAL(0, retval.ret);
  TH_EQUAL("hello\n", retval.out);
  TH_EQUAL("", retval.err);

  retval = flit::call_with_output(
      "python3 -c \"import sys; sys.stderr.write('hi')\"");
  TH_EQUAL(0, retval.ret);
  TH_EQUAL("", retval.out);
  TH_EQUAL("hi", retval.err);

  retval = flit::call_with_output("/no/such/command");
  TH_EQUAL(127, retval.ret);
  TH_EQUAL("", retval.out);
  // This is not robust as different versions output different messages
  //TH_EQUAL("sh: /no/such/command: No such file or directory\n", retval.err);
}
TH_REGISTER(tst_subprocess);

namespace mymains {

void print_main (std::ostream& out, const char* name, int argCount,
                 char* argList[])
{
  out << name << "(" << argCount << ", {";
  bool first = true;
  for (int i = 0; i < argCount; i++) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << argList[i];
  }
  out << "})\n";
}

int mymain_1 (int argCount, char* argList[]) {
  print_main(std::cout, "mymain_1", argCount, argList);
  std::cerr << "Called from mymain_1" << std::endl;
  return 0;
}
FLIT_REGISTER_MAIN(mymain_1)

int mymain_2 (int argCount, char* argList[]) {
  print_main(std::cout, "mymain_2", argCount, argList);
  std::cerr << "Called from mymain_2" << std::endl;
  std::exit(3);
}
FLIT_REGISTER_MAIN(mymain_2)

int mymain_3 (int argCount, char* argList[]) {
  return 2;
}
FLIT_REGISTER_MAIN(mymain_3)

int unregistered_main (int argCount, char* argList[]) {
  print_main(std::cout, "mymain_2", argCount, argList);
  std::cerr << "Called from mymain_2" << std::endl;
  std::exit(3);
}
// unregistered_main is not registered

}

void tst_register_main_func() {
  // just test that the previously registered ones are there
  TH_EQUAL(flit::find_main_func("mymain_1"), mymains::mymain_1);
  TH_EQUAL(flit::find_main_func("mymain_2"), mymains::mymain_2);
  TH_EQUAL(flit::find_main_func("mymain_3"), mymains::mymain_3);

  TH_EQUAL(flit::find_main_name(mymains::mymain_1), "mymain_1");
  TH_EQUAL(flit::find_main_name(mymains::mymain_2), "mymain_2");
  TH_EQUAL(flit::find_main_name(mymains::mymain_3), "mymain_3");

  // test with elements not registered
  TH_THROWS(flit::find_main_func("unregistered_main"), std::out_of_range);
  TH_THROWS(flit::find_main_name(mymains::unregistered_main),
            std::out_of_range);

  // test find_main_name with a nullptr
  TH_THROWS(flit::find_main_name(nullptr), std::invalid_argument);

  // test that we cannot add duplicate names or function pointers
  TH_THROWS(flit::register_main_func("mymain_1", mymains::unregistered_main),
            std::logic_error);
  TH_THROWS(flit::register_main_func("unregistered_main", mymains::mymain_1),
            std::logic_error);

  // test with a null pointer
  TH_THROWS(flit::register_main_func("unique", nullptr), std::invalid_argument);
}
TH_REGISTER(tst_register_main_func);

int othermain_1(int, char**) { return 0; }
int othermain_2(int, char**) { return 0; }

void tst_register_main_duplicate_func() {
  // register the main functions
  flit::register_main_func("othermain_1", othermain_1);
  flit::register_main_func("othermain_2", othermain_2);

  // test that registering already registered things are okay, as long as they
  // match.
  flit::register_main_func("othermain_1", othermain_1);
  flit::register_main_func("othermain_2", othermain_2);

  // test that registering already registered things with new mappings is an
  // error.
  TH_THROWS(flit::register_main_func("othermain_2", othermain_1),
            std::logic_error);
  TH_THROWS(flit::register_main_func("othermain_1", othermain_2),
            std::logic_error);
}

// This test sufficiently exercises isFastTrack() and callFastTrack()
// Therefore, we do not need to test those functions in isolation
void tst_call_main() {
  // this is a prerequisite to calling call_main()
  flit::g_program_path = flit::which("./tst_subprocess");

  auto result = flit::call_main(mymains::mymain_1, "arbitrary_name",
                                "the remaining args");
  TH_EQUAL(0, result.ret);
  TH_EQUAL("mymain_1(4, {arbitrary_name, the, remaining, args})\n", result.out);
  TH_EQUAL("Called from mymain_1\n", result.err);

  result = flit::call_main(mymains::mymain_2, "another name",
                           "'only one argument'");
  TH_EQUAL(3, result.ret);
  TH_EQUAL("mymain_2(2, {another name, only one argument})\n", result.out);
  TH_EQUAL("Called from mymain_2\n", result.err);

  result = flit::call_main(mymains::mymain_2, "main2", "");
  TH_EQUAL(3, result.ret);
  TH_EQUAL("mymain_2(1, {main2})\n", result.out);
  TH_EQUAL("Called from mymain_2\n", result.err);

  result = flit::call_main(mymains::mymain_3, "", "");
  TH_EQUAL(2, result.ret);
  TH_EQUAL("", result.out);
  TH_EQUAL("", result.err);

  // throws if the function was not registered.
  TH_THROWS(flit::call_main(mymains::unregistered_main, "unregistered_main",
                            "arguments"),
            std::logic_error);
}
TH_REGISTER(tst_call_main);
