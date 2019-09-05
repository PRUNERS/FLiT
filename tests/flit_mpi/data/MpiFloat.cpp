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

#include <flit.h>

#include <mpi.h>

#include <string>
#include <sstream>
#include <fstream>

// TODO: put these into FLiT?
template <typename F> MPI_Datatype mpi_type();
template <> inline MPI_Datatype mpi_type<char> () { return MPI_BYTE; }
template <> inline MPI_Datatype mpi_type<short> () { return MPI_SHORT; }
template <> inline MPI_Datatype mpi_type<int> () { return MPI_INT; }
template <> inline MPI_Datatype mpi_type<long> () { return MPI_LONG; }
template <> inline MPI_Datatype mpi_type<long long> () { return MPI_LONG_LONG; }
template <> inline MPI_Datatype mpi_type<unsigned char> () {
  return MPI_UNSIGNED_CHAR;
}
template <> inline MPI_Datatype mpi_type<unsigned short> () {
  return MPI_UNSIGNED_SHORT;
}
template <> inline MPI_Datatype mpi_type<unsigned int> () {
  return MPI_UNSIGNED;
}
template <> inline MPI_Datatype mpi_type<unsigned long> () {
  return MPI_UNSIGNED_LONG;
}
template <> inline MPI_Datatype mpi_type<unsigned long long> () {
  return MPI_UNSIGNED_LONG_LONG;
}
template <> inline MPI_Datatype mpi_type<float> () { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_type<double> () { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<long double> () {
  return MPI_LONG_DOUBLE;
}

// this is the real test, run under MPI in separate processes
template <typename F>
int mpi_main_F(int argCount, char* argList[]) {
  MPI_Init(&argCount, &argList);

  int world_size = -1;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string logfile = "out-" + std::to_string(rank) + ".log";
  std::ofstream out(logfile);

  out << "\n";
  out << "hello from rank " << rank << " of " << world_size << "\n";
  out << "mpi_main_F<" << typeid(F).name() << ">(" << argCount << ", {";
  bool first = true;
  for (int i = 0; i < argCount; i++) {
    if (!first) { out << ", "; }
    first = false;
    out << argList[i];
  }
  out << "})\n";

  // send a message from rank 0 to rank 1
  if (rank == 0) {
    F msg = 3.14159;
    MPI_Send(&msg, 1, mpi_type<F>(), 1, 0, MPI_COMM_WORLD);
    out << "Sending '" << msg<< "' from rank 0\n";
    out.flush();
  } else if (rank == 1) {
    F received = 0;
    MPI_Recv(&received, 1, mpi_type<F>(), 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    out << "Received '" << received << "' from rank 0 to rank 1\n";
  } else {
    throw std::logic_error("there should only be two ranks");
  }

  MPI_Finalize();

  return 0;
}

int mpi_main_float(int argc, char** argv) {
  return mpi_main_F<float>(argc, argv);
}

int mpi_main_double(int argc, char** argv) {
  return mpi_main_F<double>(argc, argv);
}

int mpi_main_long_double(int argc, char** argv) {
  return mpi_main_F<long double>(argc, argv);
}

FLIT_REGISTER_MAIN(mpi_main_float);
FLIT_REGISTER_MAIN(mpi_main_double);
FLIT_REGISTER_MAIN(mpi_main_long_double);

template <typename F> flit::MainFunc* get_mpi_main();
template <> inline flit::MainFunc* get_mpi_main<float>() {
  return mpi_main_float;
}
template <> inline flit::MainFunc* get_mpi_main<double>() {
  return mpi_main_double;
}
template <> inline flit::MainFunc* get_mpi_main<long double>() {
  return mpi_main_long_double;
}

template <typename T>
class MpiFloat : public flit::TestBase<T> {
public:
  MpiFloat(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { 1 };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    auto main_func = get_mpi_main<T>();
    flit::TempDir tempdir;      // create a temp dir
    flit::ProcResult result;

    // change to a different directory when calling main
    // this is so that the output files will be put into the temporary dir
    // rather than in the current directory.
    // This trick is nice because it now makes this test reentrant since each
    // invocation will executed from a separate directory.
    {
      flit::PushDir pushd(tempdir.name());  // go to it
      auto result = flit::call_mpi_main(main_func, "mpirun -n 2", "mympi",
                                        "remaining arguments");
    }

    auto logcontents_0 = flit::readfile(
        flit::join(tempdir.name(), "out-0.log"));
    auto logcontents_1 = flit::readfile(
        flit::join(tempdir.name(), "out-1.log"));

    std::vector<std::string> vec_result;
    vec_result.emplace_back(std::to_string(result.ret));
    vec_result.emplace_back(result.out);
    vec_result.emplace_back(result.err);
    vec_result.emplace_back(logcontents_0);
    vec_result.emplace_back(logcontents_1);
    return vec_result;
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(MpiFloat)
