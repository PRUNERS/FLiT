TARGET        = testinputs

CC            = g++
COMMON_FLAGS += -std=c++14
COMMON_FLAGS += -g
COMMON_FLAGS += -Werror
COMMON_FLAGS += -Wall
COMMON_FLAGS += -Wextra
COMMON_FLAGS += -Wstrict-aliasing=0
UNOPT_FLAGS   = -O0
OPT_FLAGS    += -O2
OPT_FLAGS    += -funsafe-math-optimizations
LINK_FLAGS    =
DEP_FLAGS     = -MD -MF $*.d

SOURCE        = $(wildcard *.cpp)
UNOPT_OBJ    += groundtruth.unopt.o
OPT_OBJ      += test.opt.o
OPT_OBJ      += main.opt.o
OBJ           = $(UNOPT_OBJ) $(OPT_OBJ)
DEPS          = $(SOURCE:%.cpp=%.d)

.PHONY: default
default: all

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f $(DEPS)

.PHONY: veryclean distclean
veryclean: distclean
distclean: clean
	rm -f $(TARGET)

$(TARGET): $(OBJ) makefile
	$(CC) -o $(TARGET) $(LINK_FLAGS) $(OBJ)

%.unopt.o: %.cpp makefile
	$(CC) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

%.opt.o: %.cpp makefile
	$(CC) -c $(COMMON_FLAGS) $(OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

.PRECIOUS: %.d
-include $(DEPS)
