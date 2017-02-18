
#include <stdio.h>
#include <assert.h>
#include <math.h> 
#include <float.h>
extern "C" {
#include "quadmath.h"
}


FILE * 
s3fpGetInFile (int argc, char **argv) {
  assert(argc >= 3); 
  
  char *inname = argv[argc-2]; 
  assert(inname != NULL); 
  
  FILE *infile = fopen(inname, "r"); 
  assert(infile != NULL); 
  
  return infile; 
}


FILE * 
s3fpGetOutFile (int argc, char **argv) {
  assert(argc >= 3); 

  char *outname = argv[argc-1]; 
  assert(outname != NULL); 
  
  FILE *outfile = fopen(outname, "w"); 
  assert(outfile != NULL); 

  return outfile; 
}


unsigned long 
s3fpFileSize (FILE *f) {
  assert(f != NULL); 

  unsigned long curr = ftell(f); 

  fseek(f, 0, SEEK_END); 
  unsigned long fsize = ftell(f); 

  fseek(f, curr, SEEK_SET); 
  
  return fsize; 
}


template<typename INPUTTYPE> 
void 
s3fpGetInputs (FILE *infile, 
	       unsigned long n_inputs, 
	       INPUTTYPE *iarr) {
  unsigned long fsize = s3fpFileSize(infile); 
  
  assert(fsize == n_inputs * sizeof(INPUTTYPE)); 
  
  fread(iarr, sizeof(INPUTTYPE), n_inputs, infile); 
}


template<typename FPTYPE> 
FPTYPE 
s3fpFPPerturbation (FPTYPE base, 
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
void 
s3fpWriteOutputs (FILE *outfile, 
		  unsigned int n_outputs, 
		  OUTPUTTYPE *oarr) {
  assert(outfile != NULL); 

  fwrite(oarr, sizeof(OUTPUTTYPE), n_outputs, outfile); 
}

