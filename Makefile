#CC             := clang++
CC             := g++
FFLAGS         ?=
LIBDIR         := lib
SRCDIR         := src
TARGET         ?= $(LIBDIR)/libflit.so

CPPFLAGS       += $(FFLAGS)
CPPFLAGS       += -Wuninitialized -g
CPPFLAGS       += -fPIC
CPPFLAGS       += -std=c++11
CPPFLAGS       += -Wno-shift-count-overflow
CPPFLAGS       += -Wall
CPPFLAGS       += -Wextra
CPPFLAGS       += -Werror
CPPFLAGS       += -I.

CPPFLAGS       += $(S3_REQUIRED)
LINKFLAGS      += -lm -shared

DEPFLAGS       += -MD -MF $(SRCDIR)/$*.d

SOURCE         := $(wildcard $(SRCDIR)/*.cpp)

OBJ            := $(SOURCE:.cpp=.o)
DEPS           := $(SOURCE:.cpp=.d)


.PHONY : all
all :  $(TARGET)

$(TARGET) : $(OBJ)
	mkdir -p lib
	$(CC) $(CPPFLAGS) -o $@ $^ $(LINKFLAGS)

$(SRCDIR)/%.o : $(SRCDIR)/%.cpp Makefile
	$(CC) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

.PRECIOUS: src/%.d
-include $(SOURCE:%.cpp=%.d)

.PHONY : clean
clean :
	rm -f $(OBJ)
	rm -f $(DEPS)

.PHONY: veryclean distclean
veryclean : distclean
distclean : clean
	rm -f $(TARGET)

