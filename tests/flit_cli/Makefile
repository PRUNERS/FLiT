SUB_MAKES     := $(wildcard */Makefile */makefile)
SUB_DIRS      := $(patsubst %/,%,$(dir $(SUB_MAKES)))
CHECK_TARGETS := $(addprefix check__,$(SUB_DIRS))
CLEAN_TARGETS := $(addprefix clean__,$(SUB_DIRS))
BUILD_TARGETS := $(addprefix build__,$(SUB_DIRS))

ifndef VERBOSE
.SILENT:
endif

.PHONY: build check help clean
build: $(BUILD_TARGETS)

check: $(CHECK_TARGETS)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile all subdirectories"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

.PHONY: build__% check__% clean__%

build__%:
	@$(MAKE) --no-print-directory build --directory $*

check__%:
	@$(MAKE) --no-print-directory check --directory $*

clean__%:
	@$(MAKE) --no-print-directory clean --directory $*

