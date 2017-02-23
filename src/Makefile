# a makefile for a variety of compilers and compiler flags

#CUDA
NVCC 		:= /usr/local/cuda-7.5/bin/nvcc
NVCC_VER 	:= nvcc-7.5
NVCC_REQUIRED 	+= --std=c++11
NVCC_REQUIRED 	+= -O0
NVCC_REQUIRED 	+= -ccbin=g++
DEVCAP 		?= -arch=compute_37 -code=sm_37
NVCC_REQUIRED 	+= $(DEVCAP)

NVCC_CFLAGS 	+= -I .
NVCC_CFLAGS 	+= -x cu
NVCC_CFLAGS 	+= -dc
NVCC_CFLAGS 	+= -D__CUDA__
NVCC_CFLAGS 	+= -I/usr/local/cuda-7.5/samples/common/inc

HAS_CUDA      	:= $(shell command -v nvcc 2> /dev/null)

CLANG 		:= clang++
INTEL 		:= icpc
GCC 		:= g++


ifndef CUDA_ONLY
COMPILERS := $(foreach c, GCC INTEL CLANG, $(if $(shell which $($(c))), $c,))
endif

#$(info COMPILERS=$(COMPILERS))

#gives highly detailed output (general scores aren't perfect indexes of results --
#  i.e. different results can produce identical scores
VERBOSE ?=

#for S3FP related test
S3_REQUIRED 	:= -IS3FP/
S3_REQUIRED 	+= -DWFT=float -DSCH_LIB

CC_REQUIRED	+= $(S3_REQUIRED)
NVCC_REQUIRED	+= $(S3_REQUIRED)

CC_REQUIRED  	+= -std=c++14
CC_REQUIRED 	+= -I.

LD_REQUIRED 	:= -lm
LD_REQUIRED 	+= -lstdc++


#CC_REQUIRED 	+= -lquadmath

#on systems with non-standard gcc installations (such as module), clang may
#be unable to determine the correct gcc toolchain
GCC_TOOLCHAIN	:= $(dir $(shell which g++))..
CLANG_REQUIRED	:= --gcc-toolchain=$(GCC_TOOLCHAIN)

#$(info CLANG_REQUIRED = $(CLANG_REQUIRED))

#FP FLAGS
include ../Makefile.switches

TESTS       := $(wildcard tests/*.cpp)
SOURCE      := $(wildcard *.cpp)
SOURCE      += $(TESTS)

#$(info SOURCE = $(SOURCE))
hostname    := $(shell hostname)

ifdef HAS_CUDA
CUSOURCE    += $(TESTS)
CUSOURCE    += $(wildcard *.cpp)
CUOBJ	    += $(CUSOURCE:%.cpp=%.o)
include ../Makefile.switches.cuda 
CUTARGETS := $(foreach s, $(CUSWITCHES), \
		$(NVCC_VER)_$(hostname)_$(strip $(s))_out_$(VERBOSE))
endif

TARGETS  	:= $(foreach c, $(COMPILERS), \
		$(foreach s, $(SWITCHES_$(strip $c)), \
		$(foreach o, $(OPCODES), \
		$c_$(hostname)_$(strip $(s))_$(strip $(o))_out_$(VERBOSE))))

VERBOSE_OUTPUT := $(TARGETS:%=%_out_$(VERBOSE))

.PHONY : all
all : $(TARGETS) $(CUTARGETS) cleanResults archive

# very cool: recyclable target definitions! adapted from
# http://make.mad-scientist.net/the-eval-function/

# $1 is switches, $2 is compiler, $3 is optimization level
define RULE
#run test and collect results
$(strip $2)_$(hostname)_$(strip $1)_$(strip $3)_out_$(VERBOSE) : $2_$(hostname)_$(strip $1)_$(strip $3)
	-./$$< $(VERBOSE) > $$@
	-sed -i 's/HOST/$(hostname)/g' $$@
	-sed -i 's/SWITCHES/$($(strip $1))/g' $$@
	-sed -i 's/OPTL/$($(strip $3))/g' $$@
	-sed -i 's/COMPILER/$($(strip $2))/g' $$@
	-sed -i 's/FILENAME/$2_$(hostname)_$(strip $1)_$(strip $3)/g' $$@

#link
$(strip $2)_$(hostname)_$(strip $1)_$(strip $3) : $(SOURCE:%.cpp=%_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o)
	-$($(strip $2)) $($(strip $1)) $($(strip $3)) $($(strip $2)_REQUIRED)\
	 $(CC_REQUIRED) $(LD_REQUIRED) $$^ -o $$@
	-rm $(SOURCE:.cpp=_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o)

#compile
%_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o : %.cpp
	-$($(strip $2)) -c $($(strip $1)) $($(strip $3)) $(CC_REQUIRED) \
	$($(strip $2)_REQUIRED) $$< -o $$@
endef

# $(info opcodes = $(OPCODES))
# $(info SWITCHES_CLANG = $(SWITCHES_CLANG))
# $(info SWITCHES_GCC = $(SWITCHES_GCC))
# $(info SWITCHES_INTEL = $(SWITCHES_INTEL))

$(foreach c, $(COMPILERS), $(foreach s, $(SWITCHES_$(strip $c)), \
	$(foreach o, $(OPCODES), \
	$(eval $(call RULE, $s, $c, $o)))))

define CURULE
#$1 is switches
#run test
$(NVCC_VER)_$(hostname)_$(strip $1)_out_$(VERBOSE) : $(NVCC_VER)_$(hostname)_$(strip $1)
	./$$< $(VERBOSE) > $$@
	-sed -i 's/HOST/$(hostname)/g' $$@
	-sed -i 's/SWITCHES/$($(strip $1))/g' $$@
	-sed -i 's/OPTL//g' $$@
	-sed -i 's/COMPILER/$(NVCC_VER)/g' $$@
	-sed -i 's/FILENAME/$(NVCC_VER)_$(hostname)_$(strip $1)/g' $$@

#link test
$(NVCC_VER)_$(hostname)_$(strip $1) : $(CUOBJ:%.o=%_$(NVCC_VER)_$(hostname)_$(strip $1).o)
	$(NVCC) $($(strip $1)) $(NVCC_REQUIRED) $$^ -o $$@
	rm $(CUOBJ:.o=_$(strip $(NVCC_VER))_$(hostname)_$(strip $1).o)

#compile test
%_$(strip $(NVCC_VER))_$(hostname)_$(strip $1).o : %.cpp
	-$(NVCC) -c $($(strip $1)) $(NVCC_CFLAGS) $(NVCC_REQUIRED) $$< -o $$@
endef

$(foreach s, $(CUSWITCHES), $(eval $(call CURULE, $s)))

.PHONY : cleanResults #strip excess info from Intel output (__file__) and remove emtpy outputs
cleanResults : $(TARGETS) $(CUTARGETS)
	$(info doing clean)
	sed -i 's/FPTests:://' *out_$(VERBOSE)
	#-find . -size 0 -exec rm {} \;

.PHONY : archive
archive : cleanResults
	mkdir -p ../results
	mv *$(hostname)* ../results
#	tar -zcf ../results/$(hostname)_$(shell date +%m%d%y).tgz ../results/*
	#rm -f *$(hostname)*

.PHONY : clean
clean :
	rm -f $(TARGETS)
	rm -f $(CUTARGETS)
	rm -f $(VERBOSE_OUTPUT)
	rm -f *.o tests/*.o 
	rm -f *.d tests/*.d
