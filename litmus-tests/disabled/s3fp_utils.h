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

#ifndef S3FP_UTILS_H
#define S3FP_UTILS_H

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>


inline FILE* s3fpGetInFile (int argc, char **argv) {
  assert(argc >= 3);

  char *inname = argv[argc-2];
  assert(inname != NULL);

  FILE *infile = fopen(inname, "r");
  assert(infile != NULL);

  return infile;
}


inline FILE* s3fpGetOutFile (int argc, char **argv) {
  assert(argc >= 3);

  char *outname = argv[argc-1];
  assert(outname != NULL);

  FILE *outfile = fopen(outname, "w");
  assert(outfile != NULL);

  return outfile;
}


inline unsigned long s3fpFileSize (FILE *f) {
  assert(f != NULL);

  unsigned long curr = ftell(f);

  fseek(f, 0, SEEK_END);
  unsigned long fsize = ftell(f);

  fseek(f, curr, SEEK_SET);

  return fsize;
}

template<typename INPUTTYPE>
void s3fpGetInputs (FILE *infile,
	       unsigned long n_inputs,
	       INPUTTYPE *iarr) {
  unsigned long fsize = s3fpFileSize(infile);

  assert(fsize == n_inputs * sizeof(INPUTTYPE));

  fread(iarr, sizeof(INPUTTYPE), n_inputs, infile);
}


template<typename FPTYPE>
FPTYPE s3fpFPPerturbation (FPTYPE base,
		    int range_lb,
		    int range_ub) {
  assert(range_lb <= range_ub);

  assert(sizeof(FPTYPE) == sizeof(float) ||
	 sizeof(FPTYPE) == sizeof(double));

  if (range_lb == range_ub) return base;

  int range_diff = (range_ub - range_lb) + 1;
  int shift_dis = (rand() % range_diff) + range_lb;

  if (shift_dis > 0) {
    FPTYPE ret = base;
    FPTYPE before;
    for (int i = 0 ; i < shift_dis ; i++) {
      before = ret;
      if (sizeof(FPTYPE) == sizeof(float))
	ret = nexttowardf(ret, FLT_MAX);
      else if (sizeof(FPTYPE) == sizeof(double))
	ret = nexttoward(ret, DBL_MAX);
      else assert(false);
      assert(ret != before);
    }
    return ret;
  }
  else if (shift_dis < 0) {
    FPTYPE ret = base;
    FPTYPE before;
    for (int i = 0 ; i < (-1 * shift_dis) ; i++) {
      before = ret;
      if (sizeof(FPTYPE) == sizeof(float))
	ret = nexttowardf(ret, -1 * FLT_MAX);
      else if (sizeof(FPTYPE) == sizeof(double))
	ret = nexttoward(ret, -1 * DBL_MAX);
      else assert(false);
      assert(ret != before);
    }
    return ret;
  }
  else { // shift_dis == 0
    return base;
  }
}


template<typename OUTPUTTYPE>
void s3fpWriteOutputs (FILE *outfile,
		  unsigned int n_outputs,
		  OUTPUTTYPE *oarr) {
  assert(outfile != NULL);

  fwrite(oarr, sizeof(OUTPUTTYPE), n_outputs, outfile);
}

#endif // S3FP_UTILS_H
