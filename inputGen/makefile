TARGET              = inputGen
OPT_LIB             = testbed.so

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

OBJ_DIR             = .obj
SOURCE              = $(wildcard *.cpp)
OPT_SOURCE          = testbed.cpp
UNOPT_SOURCE        = $(filter-out $(OPT_SOURCE),$(SOURCE))
TEST_SOURCE         = $(wildcard ../qfpc/tests/*.cpp)
TEST_LINKS          = $(notdir $(TEST_SOURCE))
TEST_LINKS         := $(filter-out EigenTests.cpp,$(TEST_LINKS))
OPT_SOURCE         += $(TEST_LINKS)
UNOPT_SOURCE       += $(TEST_LINKS)
UNOPT_OBJ           = $(addprefix $(OBJ_DIR)/,$(notdir $(UNOPT_SOURCE:%.cpp=%.unopt.o)))
OPT_OBJ             = $(addprefix $(OBJ_DIR)/,$(notdir $(OPT_SOURCE:%.cpp=%.opt.o)))
OBJ                 = $(UNOPT_OBJ) $(OPT_OBJ)
DEPS                = $(SOURCE:%.cpp=%.d)
DEPS               += $(TEST_LINKS:%.cpp=%.d)

print:
	@echo "OPT_SOURCE:"
	@printf "  %s\n" $(OPT_SOURCE)
	@echo "SOURCE:"
	@printf "  %s\n" $(SOURCE)
	@echo "OBJ:"
	@printf "  %s\n" $(OBJ)
	@echo "TEST_LINKS:"
	@printf "  %s\n" $(TEST_LINKS)

.PHONY: default
default: all

.PHONY: all
all: $(TARGET) $(OPT_LIB)

link: $(TEST_LINKS)

.PHONY: run
run: $(TARGET)
	-./$(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f $(DEPS)
	rm -f $(TEST_LINKS)
	-rmdir .obj

.PHONY: veryclean distclean
veryclean: distclean
distclean: clean
	rm -f $(TARGET)
	rm -f $(OPT_LIB)

$(OPT_LIB): $(OPT_OBJ) makefile
	$(CXX) -o $(OPT_LIB) -shared $(OPT_OBJ) $(LINK_FLAGS)

$(TARGET): $(UNOPT_OBJ) makefile
	$(CXX) -o $(TARGET) $(UNOPT_OBJ) $(LINK_FLAGS)

%.cpp: ../qfpc/tests/%.cpp
	ln -s $< $@

$(OBJ_DIR)/%.unopt.o: %.cpp makefile
	mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

$(OBJ_DIR)/%.opt.o: %.cpp makefile
	mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(INTERNAL_OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

.PRECIOUS: %.d
-include $(DEPS)
