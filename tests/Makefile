CC          := g++
CFLAGS      := -O2
CFLAGS      += -std=c++11
LDFLAGS     :=
TESTER_SRC  := harness_tester.cpp
HARNESS     := test_harness.h
TESTER      := $(TESTER_SRC:%.cpp=%)
SUB_MAKES   := $(wildcard */Makefile */makefile)
SUB_DIRS    := $(dir $(SUB_MAKES))
SUB_TARGETS := $(addprefix sub__,$(patsubst %/,%,$(SUB_DIRS)))

.PHONY: default check help clean run__% sub__%
default: $(TESTER)

check: run__$(TESTER) $(SUB_TARGETS)
	@echo "All tests pass"

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  default  just compile the $(TESTER)"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean:
	rm -f $(TESTER)

run__$(TESTER): $(TESTER)
	@./$(TESTER) | grep "failures: *6" >/dev/null

sub__%:
	@$(MAKE) check --directory $*

$(TESTER): $(TESTER_SRC) $(HARNESS) Makefile
	@echo "Compiling $(TESTER)"
	@$(CC) $(CFLAGS) $(TESTER_SRC) -o $(TESTER) $(LD_FLAGS)

