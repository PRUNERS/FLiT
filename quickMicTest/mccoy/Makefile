CC := icpc
FLAGS := -O2

.phony : all

all : test3.O2.mic test3.O2

test3.O2.mic : test3.cpp
	$(CC) $(FLAGS) -mmic $< -o $@

test3.O2 : test3.cpp
	$(CC) $(FLAGS) $< -o $@

.phony : clean

clean : 
	rm test3.O2 test3.O2.mic
