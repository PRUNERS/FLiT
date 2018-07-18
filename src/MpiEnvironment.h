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

#ifndef MPI_ENVIRONMENT_H
#define MPI_ENVIRONMENT_H

#include "flitHelpers.h"

#ifdef FLIT_USE_MPI
#include <mpi.h>
#endif // FLIT_USE_MPI

namespace flit{

/** A structure to encompass all necessities of the MPI environment for FLiT
 *
 * This is safe to use if MPI is disabled too.  Be sure to read the
 * documentation to understand how the behavior is different if MPI is
 * disabled.
 *
 * At its construction, this struct initializes the MPI environment, so should
 * be called as soon as possible to the beginning of the application (such as
 * at the beginning of main()).  At its destruction, it will call
 * MPI_Finalize(), meaning the lifetime of this object needs to extend beyond
 * all usages of the MPI runtime.
 */
struct MpiEnvironment {
  bool enabled;
  int rank;
  int size;

  /// If mpi is enabled, initializes MPI, else does nothing
  MpiEnvironment(int &argc, char** &argv) {
#ifdef FLIT_USE_MPI
    enabled = true;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else // not defined(FLIT_USE_MPI)
    FLIT_UNUSED(argc);
    FLIT_UNUSED(argv);
    enabled = false;
    rank = 0;
    size = 1;
#endif // FLIT_USE_MPI
  }

  /// If mpi is enabled, calls MPI_Finalize(), else does nothing
  ~MpiEnvironment() {
#ifdef FLIT_USE_MPI
    MPI_Finalize();
#endif // FLIT_USE_MPI
  }

  /// If mpi is enabled, calls MPI_Abort(), else does nothing
  void abort(int retcode) {
#ifdef FLIT_USE_MPI
    MPI_Abort(MPI_COMM_WORLD, retcode);
#else
    FLIT_UNUSED(retcode);
#endif // FLIT_USE_MPI
  }

  /// Returns true if my rank is zero (i.e. I am the root MPI process)
  bool is_root() { return rank == 0; }
};

extern MpiEnvironment *mpi;

} // end of namespace flit

#endif // MPI_ENVIRONMENT_H
