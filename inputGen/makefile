# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/main/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

TARGET              = inputGen
OPT_LIB             = testbed.so

OPT_FLAGS          ?= -O3 -funsafe-math-optimizations
INTERNAL_OPT_FLAGS  = $(OPT_FLAGS) -fPIC -g

CXX                 = g++
COMMON_FLAGS       += -std=c++14
COMMON_FLAGS       += -g
#COMMON_FLAGS       += -Werror
COMMON_FLAGS       += -Wall
COMMON_FLAGS       += -Wextra
COMMON_FLAGS       += -Wstrict-aliasing=0
COMMON_FLAGS       += -I../src
COMMON_FLAGS       += -I../src/tests
UNOPT_FLAGS         = -O0
LINK_FLAGS          = -ldl -rdynamic
DEP_FLAGS           = -MD -MF $*.d

OBJ_DIR             = .obj
SOURCE              = $(wildcard *.cpp)
OPT_SOURCE          = testbed.cpp
UNOPT_SOURCE        = $(filter-out $(OPT_SOURCE),$(SOURCE))
TEST_SOURCE         = $(wildcard ../src/tests/*.cpp)
TEST_LINKS          = $(notdir $(TEST_SOURCE))
TEST_LINKS         := $(addprefix $(OBJ_DIR)/,$(filter-out EigenTests.cpp,$(TEST_LINKS)))
OPT_SOURCE         += $(TEST_LINKS)
UNOPT_SOURCE       += $(TEST_LINKS)
UNOPT_OBJ           = $(addprefix $(OBJ_DIR)/,$(notdir $(UNOPT_SOURCE:%.cpp=%.unopt.o)))
OPT_OBJ             = $(addprefix $(OBJ_DIR)/,$(notdir $(OPT_SOURCE:%.cpp=%.opt.o)))
OBJ                 = $(UNOPT_OBJ) $(OPT_OBJ)
DEPS                = $(SOURCE:%.cpp=%.d)
DEPS               += $(TEST_LINKS:%.cpp=%.d)

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

$(OBJ_DIR)/%.cpp: ../src/tests/%.cpp
	@mkdir -p $(OBJ_DIR)
	ln -sf ../$< $@

%.unopt.o: %.cpp makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

%.opt.o: %.cpp makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(INTERNAL_OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

$(OBJ_DIR)/%.unopt.o: %.cpp makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(UNOPT_FLAGS) $(DEP_FLAGS) -o $@ $<

$(OBJ_DIR)/%.opt.o: %.cpp makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c $(COMMON_FLAGS) $(INTERNAL_OPT_FLAGS) $(DEP_FLAGS) -o $@ $<

.PRECIOUS: %.d
-include $(DEPS)

.PHONY: print
print:
	@echo "OPT_SOURCE:"
	@printf "  %s\n" $(OPT_SOURCE)
	@echo "SOURCE:"
	@printf "  %s\n" $(SOURCE)
	@echo "OBJ:"
	@printf "  %s\n" $(OBJ)
	@echo "TEST_LINKS:"
	@printf "  %s\n" $(TEST_LINKS)

