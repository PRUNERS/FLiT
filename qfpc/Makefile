# a makefile for a variety of compilers and compiler flags

#CUDA
NVCC := /usr/local/cuda-7.5/bin/nvcc
NVCC_VER := nvcc-7.5
NVCC_REQUIRED += --std=c++11
NVCC_REQUIRED += -O0
NVCC_REQUIRED += -ccbin=g++-4.9
DEVCAP ?= -arch=sm_20
NVCC_REQUIRED += $(DEVCAP)

NVCC_CFLAGS += -I .
NVCC_CFLAGS += -x cu
NVCC_CFLAGS += -dc
NVCC_CFLAGS += -D__CUDA__
NVCC_CFLAGS += -I/usr/local/cuda-7.5/samples/common/inc

HAS_CUDA      := $(shell command -v nvcc 2> /dev/null)

#the list of compilers to iterate over
#compiler specific flags may be set here
GCC := g++
#CLANG := $(if $(shell which clang++-3.6), clang++-3.6, clang++)
CLANG := clang-3.9
#PG := pgCC  #doesn't fully support c++11 (even 15.7!) so not in list
INTEL := icpc -mlong-double-80

COMPILERS := $(foreach c, GCC INTEL CLANG, $(if $(shell which $($(c))), $c,))

#gives highly detailed output (general scores aren't perfect indexes of results --
#  i.e. different results can produce identical scores
VERBOSE ?=

#CC_GEN_OPTS ?= -O0 -ggdb
CC_REQUIRED  = -std=c++14
CC_REQUIRED += -Wno-shift-count-overflow
CC_REQUIRED += -lstdc++
CC_REQUIRED += -lm
CC_REQUIRED += -lpthread
CC_REQUIRED += -pthread
CC_REQUIRED += -I../gdb/inst
CC_REQUIRED += -I.


#Eigen includes

CC_REQUIRED += -isystem eigen/build/unsupported/test
CC_REQUIRED += -isystem eigen/unsupported/test
CC_REQUIRED += -isystem eigen
CC_REQUIRED += -isystem eigen/build
CC_REQUIRED += -isystem eigen/unsupported/test/../../test
CC_REQUIRED += -isystem eigen/unsupported/test/../../unsupported
CC_REQUIRED += -isystem eigen/unsupported/test/../../Eigen
CC_REQUIRED += -isystem eigen/build/unsupported/test/../../test
CC_REQUIRED += -isystem eigen/unsupported/test/./mpreal

#FP FLAGS
include ../Makefile.switches #SWITCHES
FPFLAGS := $(foreach o, $(OPCODES), $(foreach s, $(SWITCHES), $(s)$(o)))

include ../Makefile.switches.cuda #CUSWITCHES

TESTS       := $(wildcard tests/*.cpp)
SOURCE      := $(wildcard *.cpp)
SOURCE      += $(TESTS)

# CUSOURCE    += $(wildcard tests/*.cu)
# CUSOURCE    += $(wildcard *.cu)

# CUOBJ       := $(CUSOURCE:%.cu=%.o)
CUSOURCE    += $(TESTS)
CUSOURCE    += $(wildcard *.cpp)
CUOBJ	    += $(CUSOURCE:%.cpp=%.o)

SOURCE	    += eigen/EigenTests.cpp

hostname    := $(shell hostname)

ifdef HAS_CUDA
CUTARGETS := $(foreach s, $(CUSWITCHES), \
		$(NVCC_VER)_$(hostname)_$(strip $(s))_out_$(VERBOSE))
endif

TARGETS     := $(foreach c, $(COMPILERS), \
		$(foreach s, $(SWITCHES), \
		$(foreach o, $(OPCODES), \
		$c_$(hostname)_$(strip $(s))_$(strip $(o))_out_$(VERBOSE))))


VERBOSE_OUTPUT := $(TARGETS:%=%_out_$(VERBOSE))

.PHONY : all
all : $(TARGETS) $(CUTARGETS) cleanResults archive
#all : $(CUTARGETS) cleanResults archive

# very cool: recyclable target definitions! adapted from
# http://make.mad-scientist.net/the-eval-function/

define RULE
$(strip $2)_$(hostname)_$(strip $1)_$(strip $3)_out_$(VERBOSE) : $2_$(hostname)_$(strip $1)_$(strip $3)
	-./$$< $(VERBOSE) > $$@
	-sed -i 's/HOST/$(hostname)/g' $$@
	-sed -i 's/SWITCHES/$($(strip $1))/g' $$@
	-sed -i 's/OPTL/$($(strip $3))/g' $$@
	-sed -i 's/COMPILER/$($(strip $2))/g' $$@
	-sed -i 's/FILENAME/$2_$(hostname)_$(strip $1)_$(strip $3)/g' $$@

$(strip $2)_$(hostname)_$(strip $1)_$(strip $3) : $(SOURCE:%.cpp=%_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o)
	-$($(strip $2)) $($(strip $1)) $($(strip $3)) $(CC_REQUIRED) $$^ -o $$@
	-rm $(SOURCE:.cpp=_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o)

%_$(strip $2)_$(hostname)_$(strip $1)_$(strip $3).o : %.cpp
	-$($(strip $2)) -c $($(strip $1)) $($(strip $3)) $(CC_REQUIRED) $$< -o $$@
endef

$(foreach c, $(COMPILERS), $(foreach s, $(SWITCHES), $(foreach o, $(OPCODES),\
	$(eval $(call RULE, $s, $c, $o)))))

define CURULE
$(NVCC_VER)_$(hostname)_$(strip $1)_out_$(VERBOSE) : $(NVCC_VER)_$(hostname)_$(strip $1)
	-./$$< $(VERBOSE) > $$@
	-sed -i 's/HOST/$(hostname)/g' $$@
	-sed -i 's/SWITCHES/$($(strip $1))/g' $$@
	-sed -i 's/OPTL//g' $$@
	-sed -i 's/COMPILER/$(NVCC_VER)/g' $$@
	-sed -i 's/FILENAME/$(NVCC_VER)_$(hostname)_$(strip $1)/g' $$@

$(NVCC_VER)_$(hostname)_$(strip $1) : $(CUOBJ:%.o=%_$(NVCC_VER)_$(hostname)_$(strip $1).o)
	-$(NVCC) $($(strip $1)) $(NVCC_REQUIRED) $$^ -o $$@
	-rm $(CUOBJ:.o=_$(strip $(NVCC_VER))_$(hostname)_$(strip $1).o)

%_$(strip $(NVCC_VER))_$(hostname)_$(strip $1).o : %.cpp
	-$(NVCC) -c $($(strip $1)) $(NVCC_CFLAGS) $(NVCC_REQUIRED) $$< -o $$@
endef

$(foreach s, $(CUSWITCHES), $(eval $(call CURULE, $s)))

.PHONY : cleanResults #strip excess info from Intel output (__file__) and remove emtpy outputs
cleanResults : $(TARGETS) $(CUTARGETS)
	$(info doing clean)
	sed -i 's/FPTests:://' *out_$(VERBOSE)
	-find . -size 0 -exec rm {} \;

.PHONY : archive
archive : cleanResults
	mkdir -p ../results
	-tar -zcf ../results/$(hostname).tgz *$(hostname)*
	rm -f *$(hostname)*

.PHONY : clean
clean :
	rm -f $(TARGETS)
	rm -f $(CUTARGETS)
	rm -f $(VERBOSE_OUTPUT)
