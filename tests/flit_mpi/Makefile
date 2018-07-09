RUNNER        := python3
SRC           := $(wildcard tst_*.py)
RUN_TARGETS   := $(SRC:%.py=run_%)
MPIRUN        := $(shell command -v mpirun 2>/dev/null)
MPICXX        := $(shell command -v mpic++ 2>/dev/null)

include ../color_out.mk

.PHONY: check help clean build run_%
ifeq ($(MPIRUN),)
check:
	$(call color_out,RED,Warning: mpirun is not found on your system; skipping the MPI tests)
else ifeq ($(MPICXX),)
check:
	$(call color_out,RED,Warning: mpic++ is not found on your system; skipping the MPI tests)
else
check: $(TARGETS) $(RUN_TARGETS)
endif

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile the targets"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

build:
clean:

run_% : %.py
	@$(RUNNER) $<
