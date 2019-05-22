CXX           := g++
CXXFLAGS      := -O2
CXXFLAGS      += -std=c++11
LDFLAGS       :=
TESTER_SRC    := harness_tester.cpp
HARNESS       := test_harness.h
HARNESS_PY    := test_harness.py
TESTER        := $(TESTER_SRC:%.cpp=%)
SUB_MAKES     := $(wildcard */Makefile */makefile)
SUB_DIRS      := $(patsubst %/,%,$(dir $(SUB_MAKES)))
CHECK_TARGETS := $(addprefix check__,$(SUB_DIRS))
CLEAN_TARGETS := $(addprefix clean__,$(SUB_DIRS))
BUILD_TARGETS := $(addprefix build__,$(SUB_DIRS))

.PHONY: build check help clean
build: $(TESTER) $(BUILD_TARGETS)

include color_out.mk

check: run__$(TESTER) run_tester_py $(CHECK_TARGETS)
	@$(call color_out,GREEN,All tests pass)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile the $(TESTER), and all subdirectories"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean: $(CLEAN_TARGETS)
	rm -f $(TESTER)

.PHONY: run_tester_py
run_tester_py: $(HARNESS_PY)
	@$(call color_out_noline,BROWN,  testing)
	@echo " $(HARNESS_PY)"
	@python3 -m doctest $(HARNESS_PY)

.PHONY: run__% build__% check__% clean__%
run__$(TESTER): $(TESTER)
	@$(call color_out_noline,BROWN,  testing)
	@echo " $(HARNESS)"
	@./$(TESTER) | grep "failures: *10" >/dev/null

build__%:
	@$(MAKE) build --directory $*

check__%:
	@$(MAKE) check --directory $*

clean__%:
	@$(MAKE) clean --directory $*

$(TESTER): $(TESTER_SRC) $(HARNESS) Makefile
	$(CXX) $(CXXFLAGS) $(TESTER_SRC) -o $(TESTER) $(LD_FLAGS)

