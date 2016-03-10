# Makefile for KGEN-generated kernel
iTARGET := kernel.intel
gTARGET := kernel.gcc
IMODDIR := ./intel_mod
GMODDIR := ./gcc_mod
iCODE := i
gCODE := g
iFC := ifort
gFC := gfortran
iCC := icc
gCC := gcc

I_FLAGS := -no-opt-dynamic-align -fp-model source -convert big_endian -assume byterecl -ftz -traceback -assume realloc_lhs -xHost -fpp -module $(IMODDIR) -debug all
G_FLAGS := -ffree-form -ffree-line-length-none -D__GFORTRAN__ -I./ -O2 -cpp -ggdb -J $(GMODDIR)
C_FLAGS := -ggdb -O2

ALL_OBJS := kernel_driver.$c.o int3.$c.o micro_mg_cam.$c.o wv_sat_methods.$c.o shr_spfn_mod.$c.o shr_kind_mod.$c.o micro_mg2_0.$c.o micro_mg_utils.$c.o

I_OBJS := $(foreach c, i, $(ALL_OBJS))
G_OBJS := $(foreach c, g, $(ALL_OBJS))

iLIST := iCODE iTARGET IMODDIR I_FLAGS I_OBJS iFC iCC
gLIST := gCODE gTARGET GMODDIR G_FLAGS G_OBJS gFC gCC

.PHONY: all
.PHONY: clean

all:

define RULE

build: $($(word, 5, $1))
	$($(word, 6, $1)) $($(word, 4, $1)) -o $(EXE) $^

int3.$($(word, 1, $1)).o: int3.c
	$($(word, 7, $1)) $(C_FLAGS) -c -o $@ $<

kernel_driver.$($(word, 1, $1)).o: kernel_driver.f90 micro_mg_cam.$($(word, 1, $1)).o wv_sat_methods.$($(word, 1, $1)).o shr_spfn_mod.$($(word, 1, $1)).o shr_kind_mod.$($(word, 1, $1)).o micro_mg2_0.$($(word, 1, $1)).o micro_mg_utils.$($(word, 1, $1)).o
	${FC} ${FC_FLAGS} -c -o $@ $<

micro_mg_cam.$($(word, 1, $1)).o: micro_mg_cam.F90 micro_mg2_0.$($(word, 1, $1)).o shr_kind_mod.$($(word, 1, $1)).o
	${FC} ${FC_FLAGS} -c -o $@ $<

wv_sat_methods.$($(word, 1, $1)).o: wv_sat_methods.F90 
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_spfn_mod.$($(word, 1, $1)).o: shr_spfn_mod.F90 shr_kind_mod.$($(word, 1, $1)).o
	${FC} ${FC_FLAGS} -c -o $@ $<

shr_kind_mod.$($(word, 1, $1)).o: shr_kind_mod.F90 
	${FC} ${FC_FLAGS} -c -o $@ $<

micro_mg2_0.$($(word, 1, $1)).o: micro_mg2_0.F90 micro_mg_utils.$($(word, 1, $1)).o wv_sat_methods.$($(word, 1, $1)).o shr_spfn_mod.$($(word, 1, $1)).o
	${FC} ${FC_FLAGS} -c -o $@ $<

micro_mg_utils.$($(word, 1, $1)).o: micro_mg_utils.F90 shr_spfn_mod.$($(word, 1, $1)).o
	${FC} ${FC_FLAGS} -c -o $@ $<

endef

$(foreach l, iLIST gLIST, $(eval $(call RULE, $i)))

clean:
	rm -f $(EXE) $(MODDIR)/*.mod *.o
