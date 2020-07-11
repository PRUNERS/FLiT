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

#include <flit/flit.h>

#include <mpi.h>

#include <iostream>
#include <string>
#include <sstream>

#include <cstring>
#include <cstdio>

// this is the real test, run under MPI in separate processes
int mpi_main(int argCount, char* argList[]) {
  MPI_Init(&argCount, &argList);

  int world_size = -1;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::ostringstream buffer;

  printf("\n");
  printf("hello from rank %d of %d\n", rank, world_size);
  std::fflush(stdout);

  buffer << "mpi_main(" << argCount << ", {";
  bool first = true;
  for (int i = 0; i < argCount; i++) {
    if (!first) { buffer << ", "; }
    first = false;
    buffer << argList[i];
  }
  buffer << "})\n";
  printf("%s", buffer.str().c_str());
  std::fflush(stdout);

  // send a message from rank 0 to rank 1
  if (rank == 0) {
    char message[13];
    strcpy(message, "hello world!");
    MPI_Send(message, 13, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    printf("Sending '%s' from rank 0\n", message);
    std::fflush(stdout);
  } else if (rank == 1) {
    char message[13];
    MPI_Recv(message, 13, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Received '%s' from rank 0 to rank 1\n", message);
    std::fflush(stdout);
  } else {
    throw std::logic_error("there should only be two ranks");
  }

  MPI_Finalize();

  return 0;
}
FLIT_REGISTER_MAIN(mpi_main);

template <typename T>
class MpiHello : public flit::TestBase<T> {
public:
  MpiHello(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override {
    return { 1 };
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }

protected:
  using flit::TestBase<T>::id;
};

// only implement double precision
template<>
flit::Variant MpiHello<double>::run_impl(const std::vector<double> &ti)
{
  FLIT_UNUSED(ti);
  auto result = flit::call_mpi_main(mpi_main, "mpirun -n 2", "mympi",
                                    "remaining arguments");
  std::ostringstream all;
  all << result;
  return all.str();
}

REGISTER_TYPE(MpiHello)
