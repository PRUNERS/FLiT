CC         := g++
CFLAGS     := -O2
CFLAGS     += -std=c++11
LDFLAGS    :=
TESTER_SRC := harness_tester.cpp
HARNESS    := test_harness.h
TESTER     := $(TESTER_SRC:%.cpp=%)

default: $(TESTER)

check: $(TESTER)
	@./$(TESTER) | grep "failures: *6" >/dev/null
	@echo "All tests pass"

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  default  just compile the $(TESTER)"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean:
	rm -f $(TESTER)

$(TESTER): $(TESTER_SRC) $(HARNESS) Makefile
	@echo "Compiling $(TESTER)"
	@$(CC) $(CFLAGS) $(TESTER_SRC) -o $(TESTER) $(LD_FLAGS)

