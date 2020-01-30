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

#include <flit/subprocess.h>
#include <flit/fsutil.h>
#include <flit/flitHelpers.h>

#include <string>
#include <sstream>
#include <map>

#include <cstdio>

namespace {

std::map<std::string, flit::MainFunc*>& main_name_map() {
  static std::map<std::string, flit::MainFunc*> mymap;
  return mymap;
}

std::map<flit::MainFunc*, std::string>& main_func_map() {
  static std::map<flit::MainFunc*, std::string> mymap;
  return mymap;
}

flit::ProcResult call_main_impl(
    flit::MainFunc *func, std::string run_wrap, std::string progname,
    std::string remaining_args)
{
  // calls myself
  auto funcname = flit::find_main_name(func);
  if (progname == "") {
    progname = flit::g_program_path;
  }
  std::ostringstream command_builder;
  command_builder << run_wrap << " "
                  << '"' << flit::g_program_path << '"'
                  << " --progname \"" << progname << "\""
                  << " --call-main " << funcname
                  << " " << remaining_args;
  return flit::call_with_output(command_builder.str());
}

} // end of unnamed namespace

namespace flit {

ProcResult call_with_output(const std::string &command) {
  flit::TempFile tempfile;

  // run the command
  std::string cmd = command + " 2>" + tempfile.name;
  FILE* output = popen(cmd.c_str(), "r");

  // grab stdout
  std::ostringstream outbuilder;
  const int bufsize = 256;
  char buf[bufsize];
  if (output) {
    while (!feof(output)) {
      if (fgets(buf, bufsize, output) != nullptr) {
        outbuilder << buf;
      }
    }
  }

  // wait and grab the return code
  int ret = pclose(output);
  auto err = flit::readfile(tempfile.name);

  if (WIFEXITED(ret)) {
    ret = WEXITSTATUS(ret);
  }

  return ProcResult { ret, outbuilder.str(), err };
}

std::ostream& operator<<(std::ostream& out, const ProcResult &res) {
  out << "ProcResult("
      << "ret="   << res.ret << ", "
      << "out=\"" << res.out << "\", "
      << "err=\"" << res.err << "\")";
  return out;
}

void register_main_func(const std::string &main_name, MainFunc* main_func) {
  if (main_func == nullptr) {
    throw std::invalid_argument("Main func is null");
  }
  auto tmp_func = main_name_map().find(main_name);
  if (tmp_func != main_name_map().end() && tmp_func->second != main_func) {
    throw std::logic_error("Main name already registered "
                           "to a different function: " + main_name);
  }
  auto tmp_name = main_func_map().find(main_func);
  if (tmp_name != main_func_map().end() && tmp_name->second != main_name) {
    throw std::logic_error("Main func already registered "
                           "with a different name: " + main_name
                           + " != " + tmp_name->second);
  }
  main_name_map()[main_name] = main_func;
  main_func_map()[main_func] = main_name;
}

MainFunc* find_main_func(const std::string &main_name) {
  return main_name_map().at(main_name);
}

std::string find_main_name(MainFunc *main_func) {
  if (main_func == nullptr) {
    throw std::invalid_argument("Main func is null");
  }
  return main_func_map().at(main_func);
}

ProcResult call_main(MainFunc *func, std::string progname,
                     std::string remaining_args)
{
  std::string run_wrapper = "";
  return call_main_impl(func, run_wrapper, progname, remaining_args);
}

ProcResult call_mpi_main(MainFunc *func, std::string mpirun_command,
                         std::string progname, std::string remaining_args)
{
  return call_main_impl(func, mpirun_command, progname, remaining_args);
}

} // end of namespace flit
