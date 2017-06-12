##Compiler setting targets
# taken from: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# among other places
#more comp settings, taken from here:
#https://software.intel.com/sites/default/files/article/326703/fp-control-2012-08.pdf
#see psql: qfp: switch_desc for description of flags
#also, psql: qfp: 

#individual flags
## optls

O0              := -O0
O1              := -O1
O2              := -O2
O3              := -O3

#switches

ASSOCMATH       := -fassociative-math
AVX             := -mavx
COMPTRANS       := -mp1
DEFFLAGS        :=
DISFMA          := -no-fma
ENAFMA          := -fma
FASTEXPREC      := -fexcess-precision=fast
FASTM           := -ffast-math
FINMATH         := -ffinite-math-only
FLUSHDEN        := -ftz
FMAGCC          := -mavx2 -mfma
FMAICC          := -march=core-avx2
FORTRULES       := -fcx-fortran-rules
FPCONT          := -ffp-contract=on
FPMODDBL        := -fp-model=double
FPMODEXC        := -fp-model=except
FPMODEXT        := -fp-model=extended
FPMODFST1       := -fp-model fast=1
FPMODFST2       := -fp-model fast=2
FPMODPRE        := -fp-model=precise
FPMODSRC        := -fp-model=source
FPMODSTR        := -fp-model=strict
FPTRAP          := -fp-trap=common
FSTORE          := -ffloat-store
LIMITEDRANGE    := -fcx-limited-range
MCONSTS         := -fmerge-all-constants
NOFLUSHDEN      := -no-ftz
NOPRECDIV       := -no-prec-div
NOTRAP          := -fno-trapping-math
PRECDIV         := -prec-div
RECIPMATH       := -freciprocal-math 
ROUNDINGMATH    := -frounding-math
ROUNDUSR        := -fp-port
SIGNALNAN       := -fsignaling-nans
SINGLEPRECCONST := -fsingle-precision-constant
SSE             := -mfpmath=sse -mtune=native
STDEXPREC       := -fexcess-precision=standard
UNSOPTS         := -funsafe-math-optimizations
USEFASTM        := --use_fast_math

# Collections 

OPCODES         := O0 O1 O2 O3

#NOTE: gcc disables ASSOCMATH @ O0
SWITCHES_GCC    += ASSOCMATH
SWITCHES_GCC    += AVX
SWITCHES_GCC    += DEFFLAGS
SWITCHES_GCC    += FASTEXPREC
SWITCHES_GCC    += FINMATH
SWITCHES_GCC    += FMAGCC
SWITCHES_GCC    += FORTRULES
SWITCHES_GCC    += FPCONT
SWITCHES_GCC    += FSTORE
SWITCHES_GCC    += LIMITEDRANGE
SWITCHES_GCC    += MCONSTS
SWITCHES_GCC    += NOTRAP
SWITCHES_GCC    += RECIPMATH
SWITCHES_GCC    += ROUNDINGMATH
SWITCHES_GCC    += SIGNALNAN
SWITCHES_GCC    += SSE
SWITCHES_GCC    += UNSOPTS

#NOTE: Clang not honoring ASSOCMATH (issues warning with 3.9)
# see: https://llvm.org/bugs/show_bug.cgi?id=27372

SWITCHES_CLANG  += ASSOCMATH
SWITCHES_CLANG  += AVX
SWITCHES_CLANG  += DEFFLAGS
SWITCHES_CLANG  += FASTEXPREC
SWITCHES_CLANG  += FINMATH
SWITCHES_CLANG  += FMAGCC
SWITCHES_CLANG  += FMAICC
SWITCHES_CLANG  += FPCONT
SWITCHES_CLANG  += FSTORE
SWITCHES_CLANG  += MCONSTS
SWITCHES_CLANG  += NOTRAP
SWITCHES_CLANG  += RECIPMATH
SWITCHES_CLANG  += ROUNDINGMATH
SWITCHES_CLANG  += SIGNALNAN
SWITCHES_CLANG  += SINGLEPRECCONST
SWITCHES_CLANG  += SSE
SWITCHES_CLANG  += STDEXPREC
SWITCHES_CLANG  += UNSOPTS

SWITCHES_INTEL  += AVX
SWITCHES_INTEL  += COMPTRANS
SWITCHES_INTEL  += DEFFLAGS
SWITCHES_INTEL  += DISFMA
SWITCHES_INTEL  += ENAFMA
SWITCHES_INTEL  += FLUSHDEN
SWITCHES_INTEL  += FMAGCC
SWITCHES_INTEL  += FMAICC
SWITCHES_INTEL  += FPMODDBL
SWITCHES_INTEL  += FPMODEXT
SWITCHES_INTEL  += FPMODFST1
SWITCHES_INTEL  += FPMODFST2
SWITCHES_INTEL  += FPMODPRE
SWITCHES_INTEL  += FPMODSRC
SWITCHES_INTEL  += FPMODSTR
SWITCHES_INTEL  += FSTORE
SWITCHES_INTEL  += LIMITEDRANGE
SWITCHES_INTEL  += MCONSTS
SWITCHES_INTEL  += NOFLUSHDEN
SWITCHES_INTEL  += NOPRECDIV
SWITCHES_INTEL  += PRECDIV
SWITCHES_INTEL  += ROUNDINGMATH
SWITCHES_INTEL  += ROUNDUSR
SWITCHES_INTEL  += SINGLEPRECCONST
SWITCHES_INTEL  += SSE
SWITCHES_INTEL  += USEFASTM

