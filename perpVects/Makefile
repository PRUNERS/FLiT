CC := g++
INTEL := icpc
CLANG := clang++
VERBOSE ?=

CC_OPTS ?= -O0 -ggdb -Wuninitialized
CC_REQUIRED := -std=c++11  -lstdc++ -lm -Wno-shift-count-overflow
##Targets
SSE := -mfpmath=sse -mtune=native
RELAXED := -funsafe-math-optimizations
hostname := $(shell hostname)

.phony : all

all : $(CC)_$(hostname)_vanilla_out $(CC)_$(hostname)_sse_out $(CC)_$(hostname)_relaxed_out $(INTEL)_$(hostname)_vanilla_out $(INTEL)_$(hostname)_sse_out $(INTEL)_$(hostname)_relaxed_out $(CLANG)_$(hostname)_vanilla_out $(CLANG)_$(hostname)_sse_out $(CLANG)_$(hostname)_relaxed_out

#GCC

$(CC)_$(hostname)_vanilla_out : $(CC)_$(hostname)_vanilla
	./$< $(VERBOSE) > $@

$(CC)_$(hostname)_sse_out : $(CC)_$(hostname)_sse
	./$< $(VERBOSE) > $@

$(CC)_$(hostname)_relaxed_out : $(CC)_$(hostname)_relaxed
	./$< $(VERBOSE) > $@

$(CC)_$(hostname)_vanilla : test.cpp
	$(CC) $(CC_OPTS) $(CC_REQUIRED) $< -o $@

$(CC)_$(hostname)_sse : test.cpp
	$(CC) $(CC_OPTS) $(CC_REQUIRED) $(SSE) $< -o $@

$(CC)_$(hostname)_relaxed : test.cpp
	$(CC) $(CC_OPTS) $(CC_REQUIRED) $(RELAXED) $< -o $@


#ICPC

$(INTEL)_$(hostname)_vanilla_out : $(INTEL)_$(hostname)_vanilla
	./$< $(VERBOSE) > $@

$(INTEL)_$(hostname)_sse_out : $(INTEL)_$(hostname)_sse
	./$< $(VERBOSE) > $@

$(INTEL)_$(hostname)_relaxed_out : $(INTEL)_$(hostname)_relaxed
	./$< $(VERBOSE) > $@

$(INTEL)_$(hostname)_vanilla : test.cpp
	$(INTEL) $(INTEL_OPTS) $(CC_REQUIRED) $< -o $@

$(INTEL)_$(hostname)_sse : test.cpp
	$(INTEL) $(CC_OPTS) $(CC_REQUIRED) $(SSE) $< -o $@

$(INTEL)_$(hostname)_relaxed : test.cpp
	$(INTEL) $(CC_OPTS) $(CC_REQUIRED) $(RELAXED) $< -o $@


#CLANG

$(CLANG)_$(hostname)_vanilla_out : $(CLANG)_$(hostname)_vanilla
	./$< $(VERBOSE) > $@

$(CLANG)_$(hostname)_sse_out : $(CLANG)_$(hostname)_sse
	./$< $(VERBOSE) > $@

$(CLANG)_$(hostname)_relaxed_out : $(CLANG)_$(hostname)_relaxed
	./$< $(VERBOSE) > $@

$(CLANG)_$(hostname)_vanilla : test.cpp
	$(CLANG) $(CC_OPTS) $(CC_REQUIRED) $< -o $@

$(CLANG)_$(hostname)_sse : test.cpp
	$(CLANG) $(CC_OPTS) $(CC_REQUIRED) $(SSE) $< -o $@

$(CLANG)_$(hostname)_relaxed : test.cpp
	$(CLANG) $(CC_OPTS) $(CC_REQUIRED) $(RELAXED) $< -o $@



.phony : clean

clean :
	-rm $(CC)* $(CLANG)* $(INTEL)*
