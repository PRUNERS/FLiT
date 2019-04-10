SUB_MAKES     := $(wildcard */Makefile */makefile)
SUB_DIRS      := $(patsubst %/,%,$(dir $(SUB_MAKES)))
CHECK_TARGETS := $(addprefix check__,$(SUB_DIRS))
CLEAN_TARGETS := $(addprefix clean__,$(SUB_DIRS))
BUILD_TARGETS := $(addprefix build__,$(SUB_DIRS))

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
	@$(MAKE) build --directory $*

check__%:
	@$(MAKE) check --directory $*

clean__%:
	@$(MAKE) clean --directory $*

