TARGET        = testinputs

OPT_FLAGS    ?= -O3 -funsafe-math-optimizations
OPT_FLAGS    += -fPIC

CXX           = g++
COMMON_FLAGS += -std=c++14
COMMON_FLAGS += -g
COMMON_FLAGS += -Werror
COMMON_FLAGS += -Wall
COMMON_FLAGS += -Wextra
COMMON_FLAGS += -Wstrict-aliasing=0
COMMON_FLAGS += -I../qfpc
COMMON_FLAGS += -I../qfpc/tests
UNOPT_FLAGS   = -O0
LINK_FLAGS    = -ldl -rdynamic
DEP_FLAGS     = -MD -MF $*.d

SOURCE        = $(wildcard *.cpp)
SOURCE       += ../qfpc/QFPHelpers.cpp
SOURCE       += ../qfpc/InfoStream.cpp
SOURCE       += ../qfpc/testBase.cpp
OPT_SOURCE    = testbed.cpp
OPT_LIB       = testbed.so
UNOPT_SOURCE  = $(filter-out $(OPT_SOURCE),$(SOURCE))
UNOPT_OBJ     = $(UNOPT_SOURCE:%.cpp=%.unopt.o)
OPT_OBJ       = $(OPT_SOURCE:%.cpp=%.opt.o)
OBJ           = $(UNOPT_OBJ)
DEPS          = $(SOURCE:%.cpp=%.d)

.PHONY: default
default: all

.PHONY: all
all: $(TARGET) $(OPT_LIB)

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
	rm -f $(OPT_LIB)

$(OPT_LIB): $(OPT_OBJ) makefile
	$(CXX) -o $(OPT_LIB) -shared $(OPT_OBJ) $(LINK_FLAGS)

$(TARGET): $(OBJ) makefile
	$(CXX) -o $(TARGET) $(OBJ) $(LINK_FLAGS)

%.unopt.o: %.cpp makefile
	$(CXX) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

%.opt.o: %.cpp makefile
	$(CXX) -c $(COMMON_FLAGS) $(OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

.PRECIOUS: %.d
-include $(DEPS)
