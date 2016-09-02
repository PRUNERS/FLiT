TARGET              = inputGen

OPT_FLAGS          ?= -O3 -funsafe-math-optimizations
INTERNAL_OPT_FLAGS  = $(OPT_FLAGS) -fPIC

CXX                 = g++
COMMON_FLAGS       += -std=c++14
COMMON_FLAGS       += -g
COMMON_FLAGS       += -Werror
COMMON_FLAGS       += -Wall
COMMON_FLAGS       += -Wextra
COMMON_FLAGS       += -Wstrict-aliasing=0
COMMON_FLAGS       += -I../qfpc
COMMON_FLAGS       += -I../qfpc/tests
UNOPT_FLAGS         = -O0
LINK_FLAGS          = -ldl -rdynamic
DEP_FLAGS           = -MD -MF $*.d

SOURCE              = $(wildcard *.cpp)
OPT_SOURCE          = testbed.cpp
OPT_LIB             = testbed.so
UNOPT_SOURCE        = $(filter-out $(OPT_SOURCE),$(SOURCE))
UNOPT_OBJ           = $(UNOPT_SOURCE:%.cpp=%.unopt.o)
OPT_OBJ             = $(OPT_SOURCE:%.cpp=%.opt.o)
OBJ                 = $(UNOPT_OBJ) $(OPT_OBJ)
DEPS                = $(SOURCE:%.cpp=%.d)

.PHONY: default
default: all

.PHONY: all
all: $(TARGET)

.PHONY: run
run: $(TARGET)
	-./$(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f $(DEPS)

.PHONY: veryclean distclean
veryclean: distclean
distclean: clean
	rm -f $(TARGET)

$(TARGET): $(UNOPT_OBJ) makefile
	$(CXX) -o $(TARGET) $(UNOPT_OBJ) $(LINK_FLAGS)

%.unopt.o: %.cpp makefile
	$(CXX) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

%.opt.o: %.cpp makefile
	$(CXX) -c $(COMMON_FLAGS) $(INTERNAL_OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

.PRECIOUS: %.d
-include $(DEPS)
