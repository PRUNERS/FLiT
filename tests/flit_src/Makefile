CC          := g++
FLITDIR     := ../..
FLITLIBDIR  := $(FLITDIR)/lib
FLITLIB     := $(FLITLIBDIR)/libflit.so

CFLAGS      := -O0 -g
CFLAGS      += -std=c++11
LDFLAGS     := -L$(FLITLIBDIR) -lflit
LDFLAGS     += -Wl,-rpath=$(FLITLIBDIR)
SRC         := $(wildcard *.cpp)
DEPS        := $(SRC:%.cpp=%.d)
TARGETS     := $(SRC:%.cpp=%)
RUN_TARGETS := $(TARGETS:%=run_%)
HARNESS     := ../test_harness.h
CFLAGS      += $(addprefix -I,$(dir $(HARNESS)))
CFLAGS      += -I../../src

DEPFLAGS     = -MD -MF $*.d

include ../color_out.mk

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

run_% : %
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	@./$< --quiet

% : %.cpp $(HARNESS) Makefile
	$(CC) $(CFLAGS) $(DEPFLAGS) $< -o $@ $(LDFLAGS)

.PRECIOUS: %.d
-include $(DEPS)

$(TARGETS): $(FLITLIB)

$(FLITLIB):
	@$(MAKE) -C $(FLITDIR)
