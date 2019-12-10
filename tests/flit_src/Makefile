CC          := g++
FLITDIR     := ../..
FLITLIBDIR  := $(FLITDIR)/lib
FLITLIB     := $(FLITLIBDIR)/libflit.so

CFLAGS      := -O0 -g
CFLAGS      += -std=c++11
CFLAGS      += -fprofile-arcs
CFLAGS      += -ftest-coverage
CFLAGS      += -fPIC
LDFLAGS     := -L$(FLITLIBDIR) -lflit
LDFLAGS     += -Wl,-rpath=$(FLITLIBDIR)
SRC         := $(wildcard *.cpp)
DEPS        := $(SRC:%.cpp=%.d)
TARGETS     := $(SRC:%.cpp=%)
GCOV_FILES  := $(SRC:%.cpp=%.gcda)
GCOV_FILES  += $(SRC:%.cpp=%.gcno)
RUN_TARGETS := $(TARGETS:%=run_%)
HARNESS     := ../test_harness.h
CFLAGS      += $(addprefix -I,$(dir $(HARNESS)))
CFLAGS      += -I../../src

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

run_% : %
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	./$< --semi-quiet

% : %.cpp $(HARNESS) Makefile
	@$(call color_out,CYAN,  $< -> $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $< -o $@ $(LDFLAGS)

.PRECIOUS: %.d
-include $(DEPS)

$(TARGETS): $(FLITLIB)

$(FLITLIB):
	@$(MAKE) -C $(FLITDIR)
