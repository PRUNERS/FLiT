CXX         := g++
FLITDIR     := ../..
FLIT_SRC    := $(FLITDIR)/src/flit/ALL-FLIT.cpp

CXXFLAGS    := -O0 -g
CXXFLAGS    += -std=c++11
CXXFLAGS    += -fprofile-arcs
CXXFLAGS    += -ftest-coverage
CXXFLAGS    += -fPIC
LDFLAGS     := -lflit -L. -Wl,-rpath=$(CURDIR)
SRC         := $(wildcard *.cpp)
DEPS        := $(SRC:%.cpp=%.d)
DEPS        += $(notdir $(FLIT_SRC:%.cpp=%.d))
TARGETS     := $(SRC:%.cpp=%)
GCOV_FILES  := $(SRC:%.cpp=%.gcda)
GCOV_FILES  += $(SRC:%.cpp=%.gcno)
GCOV_FILES  += $(notdir $(FLIT_SRC:%.cpp=%.gcda))
GCOV_FILES  += $(notdir $(FLIT_SRC:%.cpp=%.gcno))
RUN_TARGETS := $(TARGETS:%=run_%)
HARNESS     := ../test_harness.h
CXXFLAGS    += $(addprefix -I,$(dir $(HARNESS)))
CXXFLAGS    += -I../../src

DEPFLAGS     = -MD -MF $*.d

include ../color_out.mk

ifndef VERBOSE
.SILENT:
endif

.PHONY: build check help clean run_%
build: $(TARGETS)

check: $(TARGETS) $(RUN_TARGETS)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile the targets"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean:
	rm -f $(DEPS)
	rm -f $(TARGETS)
	rm -f $(GCOV_FILES)
	rm -f libflit.so

run_% : %
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	./$< --semi-quiet

% : %.cpp $(HARNESS) Makefile
	@$(call color_out,CYAN,  $< -> $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $< -o $@ $(LDFLAGS)

.PRECIOUS: %.d
-include $(DEPS)

$(TARGETS): libflit.so

libflit.so: $(FLIT_SRC) Makefile
	@$(call color_out,CYAN,  $< -> $@)
	$(CXX) -shared $(CXXFLAGS) $(DEPFLAGS) $< -o $@
