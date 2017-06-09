#CC             := clang++
CC             := g++
#CC             := icpc
FFLAGS         ?=
TARGET         ?= runTests

CC_REQUIRED    += $(FFLAGS)
CC_REQUIRED    += -Wuninitialized -g
CC_REQUIRED    += -std=c++11
CC_REQUIRED    += -Wno-shift-count-overflow
CC_REQUIRED    += -Wall
CC_REQUIRED    += -Wextra
CC_REQUIRED    += -I.
CC_REQUIRED    += -DWFT=float -DSCH_LIB
CC_REQUIRED    += -I../src

FLITLIBDIR     := ../lib

LD_REQUIRED    += -lm
LD_REQUIRED    += -L$(FLITLIBDIR) -lflit
LD_REQUIRED    += -Wl,-rpath=$(realpath $(FLITLIBDIR))

DEPFLAGS       += -MD -MF $*.d

TESTS          := $(wildcard tests/*.cpp)
SOURCE         := $(wildcard *.cpp)
SOURCE         += $(TESTS)

OBJ            := $(SOURCE:.cpp=.o)
DEPS           := $(SOURCE:.cpp=.d)


.PHONY : all
all :  $(TARGET)

$(TARGET) : $(OBJ) Makefile
	$(CC) $(CC_REQUIRED) -o $@ $(OBJ) $(LD_REQUIRED)

%.o : %.cpp Makefile
	$(CC) $(CC_REQUIRED) $(DEPFLAGS) -c $< -o $@

.PRECIOUS: %.d
-include $(SOURCE:%.cpp=%.d)

.PHONY : clean
clean :
	rm -f $(OBJ)
	rm -f $(DEPS)

.PHONY: veryclean distclean
veryclean : distclean
distclean : clean
	rm -f $(TARGET)

-include custom.mk
