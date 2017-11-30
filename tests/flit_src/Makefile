CC          := g++
CFLAGS      := -O2
CFLAGS      += -std=c++11
LDFLAGS     :=
SRC         := $(wildcard *.cpp)
TARGETS     := $(SRC:%.cpp=%)
RUN_TARGETS := $(TARGETS:%=run_%)
HARNESS     := ../test_harness.h
CFLAGS      += $(addprefix -I,$(dir $(HARNESS)))

.PHONY: default check help clean run_%
default: $(TARGETS)

check: $(TARGETS) $(RUN_TARGETS)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  default  just compile the $(TESTER)"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean:
	rm -f $(TARGETS)

run_% : %
	@./$< --quiet

% : %.cpp $(HARNESS) Makefile
	@$(CC) $(CFLAGS) $< -o $@ $(LD_FLAGS)

