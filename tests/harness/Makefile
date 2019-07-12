CC          := g++
CFLAGS      := -O0 -g
CFLAGS      += -std=c++11
SRC         := $(wildcard *.cpp)
DEPS        := $(SRC:%.cpp=%.d)
TARGETS     := $(SRC:%.cpp=%)
RUN_TARGETS := $(TARGETS:%=run_%)
HARNESS     := $(wildcard ../*.h)
CFLAGS      += $(addprefix -I,$(sort $(dir $(HARNESS))))
TST_HARNESS := tst_test_harness

DEPFLAGS     = -MD -MF $*.d

include ../color_out.mk

ifndef VERBOSE
.SILENT:
endif

.PHONY: build check help clean run_%
build: $(TARGETS)

check: $(TARGETS) $(RUN_TARGETS)

help:
	@echo "Makefile for running tests on FLiT framework"
	@echo "  help     print this help documentation and exit"
	@echo "  build    just compile the targets"
	@echo "  check    run tests and print results to the console"
	@echo "  clean    remove all generated files"

clean:
	rm -f $(DEPS)
	rm -f $(TARGETS)

# Have a separate rule for test_harness since we expected exactly 10 failures
run_tst_test_harness: tst_test_harness
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	./tst_test_harness | grep "failures: *10$$" >/dev/null

run_% : %
	@$(call color_out_noline,BROWN,  running)
	@echo " $<"
	./$< --semi-quiet

% : %.cpp $(HARNESS) Makefile
	@$(call color_out,CYAN,  $< -> $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $< -o $@ $(LDFLAGS)

.PRECIOUS: %.d
-include $(DEPS)

$(TARGETS): $(FLITLIB)

$(FLITLIB):
	@$(MAKE) -C $(FLITDIR)
