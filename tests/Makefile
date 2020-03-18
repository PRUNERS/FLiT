CC            := g++
CFLAGS        := -O2
CFLAGS        += -std=c++11
LDFLAGS       :=
HARNESS       := test_harness.h
HARNESS_PY    := test_harness.py
SUB_MAKES     := $(wildcard */Makefile */makefile)
SUB_DIRS      := $(patsubst %/,%,$(dir $(SUB_MAKES)))
CHECK_TARGETS := $(addprefix check__,$(SUB_DIRS))
CLEAN_TARGETS := $(addprefix clean__,$(SUB_DIRS))
BUILD_TARGETS := $(addprefix build__,$(SUB_DIRS))

.PHONY: build check help clean
build: $(BUILD_TARGETS)

include color_out.mk

ifndef VERBOSE
.SILENT:
endif

check: run_tester_py $(CHECK_TARGETS)
	@$(call color_out,GREEN,All tests pass)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile all subdirectories"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean: $(CLEAN_TARGETS)

.PHONY: run_tester_py
run_tester_py: $(HARNESS_PY)
	@$(call color_out_noline,BROWN,  testing)
	@echo " $(HARNESS_PY)"
	python3 -m doctest $(HARNESS_PY)

build__%:
	@$(MAKE) --no-print-directory build --directory $*

check__%:
	@$(MAKE) --no-print-directory check --directory $*

clean__%:
	@$(MAKE) --no-print-directory clean --directory $*

