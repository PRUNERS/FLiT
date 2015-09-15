CC := g++
CC_FLAGS ?= -O0 -ggdb -Wuninitialized

.phony : all

all: test.cpp
	$(CC) $(CC_FLAGS) -std=c++11  -lstdc++ -lm ./test.cpp 
