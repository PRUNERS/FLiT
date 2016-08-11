TARGET        = testinputs

CC            = g++
COMMON_FLAGS  = -std=c++14 -g
UNOPT_FLAGS   = -O0
OPT_FLAGS     = -O2 -funsafe-math-optimizations
LINK_FLAGS    =

UNOPT_OBJ     = groundtruth.unopt.o
OPT_OBJ       = main.opt.o
OBJ           = $(UNOPT_OBJ) $(OPT_OBJ)

.PHONY: default
default: all

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJ)

.PHONY: veryclean distclean
veryclean: distclean
distclean: clean
	rm -f $(TARGET)

$(TARGET): $(OBJ) makefile
	$(CC) -o $(TARGET) $(LINK_FLAGS) $(OBJ)

%.unopt.o: %.cpp $(wildcard *.h) makefile
	$(CC) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) -o $@ $<

%.opt.o: %.cpp $(wildcard *.h) makefile
	$(CC) -c $(COMMON_FLAGS) $(OPT_FLAGS) -o $@ $<
