RUNNER        := python3
SRC           := $(wildcard tst_*.py)
RUN_TARGETS   := $(SRC:%.py=run_%)

IS_SCANBUILD  := $(shell if python3 -c 'import libscanbuild' 2>/dev/null; then \
                           echo true; \
                         fi)

include ../../color_out.mk

.PHONY: check help clean build run_%
ifeq ($(IS_SCANBUILD),true)
check: $(TARGETS) $(RUN_TARGETS)
else
check:
	$(call color_out,RED,Warning: scan-build is not found on your system;\
	  skipping flit-capture tests)
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
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	@$(RUNNER) $<
