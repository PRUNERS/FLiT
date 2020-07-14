RUNNER        := python3
SRC           := $(wildcard tst_*.py)
RUN_TARGETS   := $(SRC:%.py=run_%)

include ../../color_out.mk

ifndef VERBOSE
.SILENT:
endif

.PHONY: check help clean build run_%
check: $(TARGETS) $(RUN_TARGETS)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile the targets"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

build:
clean:

run_% : %.py
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	$(RUNNER) $<
