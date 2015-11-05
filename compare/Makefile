#this is the heart of test comparison.  
# Specifying options are filters; by not specifying an option,
# except for compiler switches (which will just use blank switches, i.e. defaults), all options will be used.
# To use it, do something like this:
# make TEST=testid COMPSW1=switchid COMPSW2=switchid SORT=[bi|lt|gt|ns] \
# PRECISION=[f|d|e] COMPILER1=[icpc|g++|clang++] COMPILER2=[icpc|g++|clang++]
# where: testid is one of:
#  0: DoOrthoPerturbTestP2
#  1: DoOrthoPerturbTestN2
#  2: DoMatrixMultSanity
#  3: DoSimpleRotate90
#  4: RotateAndUnrotate
#  5: RotateFullCircle
#  6: DoSkewSymPRotationTest
#  7: DoHariGSBasic
#  8: DoHariGSImproved

# switchid is one of the objects listed in ../Makefile.switches
# reduction SORT identifiers are self explanitory (i.e. gt = greater than)
# PRECISION is f = float, d = double, e = extended (80 bit)

include ../Makefile.switches

CXX_REQ := -std=c++11

.phony : all diff clean

all : diff

diff : run1 run2
	emacs --eval="(ediff-files \"run1\" \"run2\")"

run1 : 	test1 
	SORT=$(SORT)
	PRECISION=$(PRECISION)
	TEST=$(TEST)
	./$< verbose > $@

run2 : 	test2
	SORT=$(SORT)
	PRECISION=$(PRECISION)
	TEST=$(TEST)
	./$< verbose > $@

test1 : ../perpVects/test.cpp
	$(COMPILER1) $(CXX_REQ) $(COMPSW1) $< -o $@

test2 : ../perpVects/test.cpp
	$(COMPILER2) $(CXX_REQ) $(COMPSW2) $< -o $@

clean:
	-rm test* run*
