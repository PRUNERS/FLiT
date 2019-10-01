PREFIX         ?= /usr
DESTDIR        ?=

#CXX            ?= clang++
CXX            ?= g++
RM             ?= rm -f
RMDIR          ?= rm -rf
TARGET         ?= $(LIBDIR)/libflit.so
LIBDIR         := lib
SRCDIR         := src

CXXFLAGS       += -Wuninitialized -g
CXXFLAGS       += -fPIC
CXXFLAGS       += -std=c++11
CXXFLAGS       += -Wno-shift-count-overflow
CXXFLAGS       += -Wall
CXXFLAGS       += -Wextra
CXXFLAGS       += -Werror
CXXFLAGS       += -I.

LDFLAGS        += -shared
LDLIBS         += -lm

DEPFLAGS       += -MD -MF $(SRCDIR)/$*.d

SOURCE         := $(wildcard $(SRCDIR)/*.cpp)
HEADERS        += $(wildcard $(SRCDIR)/*.h)

OBJ            := $(SOURCE:.cpp=.o)
DEPS           := $(SOURCE:.cpp=.d)

# Install variables

SCRIPT_DIR     := scripts/flitcli
DATA_DIR       := data
CONFIG_DIR     := $(SCRIPT_DIR)/config
DOC_DIR        := documentation
LITMUS_TESTS   += $(wildcard litmus-tests/tests/*.cpp)
LITMUS_TESTS   += $(wildcard litmus-tests/tests/*.h)

EFFECTIVE_PREFIX   := $(DESTDIR)$(PREFIX)
INST_BINDIR        := $(EFFECTIVE_PREFIX)/bin
INST_LIBDIR        := $(EFFECTIVE_PREFIX)/lib
INST_INCLUDEDIR    := $(EFFECTIVE_PREFIX)/include/flit
INST_SHAREDIR      := $(EFFECTIVE_PREFIX)/share/flit
INST_LICENSEDIR    := $(EFFECTIVE_PREFIX)/share/licenses/flit
INST_FLIT_CONFIG   := $(EFFECTIVE_PREFIX)/share/flit/scripts/flitconfig.py

CAT            := $(if $(filter $(OS),Windows_NT),type,cat)
VERSION        := $(shell $(CAT) $(CONFIG_DIR)/version.txt)

-include tests/color_out.mk

# Be silent by default
ifndef VERBOSE
.SILENT:
endif

.PHONY : all
all: $(TARGET)

.PHONY: help
help:
	@echo "FLiT is an automation and analysis tool for reproducibility of"
	@echo "floating-point algorithms with respect to compilers, architectures,"
	@echo "and compiler flags."
	@echo
	@echo "The following targets are available:"
	@echo
	@echo "  all         Compiles the target $(TARGET)"
	@echo "  help        Shows this help message and exits"
	@echo "  install     Installs FLiT.  You may override the PREFIX variable"
	@echo "              to install to a different directory.  The default"
	@echo "              PREFIX value is /usr."
	@echo '                exe: "make install PREFIX=$$HOME/installs/usr"'
	@echo "  check       Run tests for FLiT framework (requires $(TARGET))"
	@echo "  clean       Clean the intermediate build artifacts from building"
	@echo "              $(TARGET)"
	@echo "  distclean   Run clean and then also remove $(TARGET)"
	@echo "  veryclean   An alias for distclean"
	@echo

$(TARGET): $(OBJ)
	@$(call color_out_noline,CYAN,  mkdir)
	@echo " lib"
	mkdir -p lib
	@$(call color_out,BLUE,Building $(TARGET))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp Makefile
	@$(call color_out,CYAN,  $< -> $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

.PRECIOUS: src/%.d
-include $(SOURCE:%.cpp=%.d)

check: $(TARGET)
	$(MAKE) check --directory tests

.PHONY: clean
clean:
	$(RM) $(OBJ)
	$(RM) $(DEPS)
	$(MAKE) clean --directory tests

.PHONY: veryclean distclean
veryclean: distclean
distclean: clean
	$(RM) $(TARGET)
	$(RMDIR) $(LIBDIR)

.PHONY: install
install: $(TARGET)
	@$(call color_out,BLUE,Installing...)
	mkdir -m 0755 -p $(INST_BINDIR)
	mkdir -m 0755 -p $(INST_LIBDIR)
	mkdir -m 0755 -p $(INST_INCLUDEDIR)
	mkdir -m 0755 -p $(INST_SHAREDIR)/scripts
	mkdir -m 0755 -p $(INST_SHAREDIR)/doc
	mkdir -m 0755 -p $(INST_SHAREDIR)/data/tests
	mkdir -m 0755 -p $(INST_SHAREDIR)/data/db
	mkdir -m 0755 -p $(INST_SHAREDIR)/config
	mkdir -m 0755 -p $(INST_SHAREDIR)/litmus-tests
	mkdir -m 0755 -p $(INST_SHAREDIR)/benchmarks
	mkdir -m 0755 -p $(INST_LICENSEDIR)
	ln -sf ../share/flit/scripts/flit.py $(INST_BINDIR)/flit
	install -m 0755 $(TARGET) $(INST_LIBDIR)/$(notdir $(TARGET))
	install -m 0644 $(HEADERS) $(INST_INCLUDEDIR)
	install -m 0755 $(SCRIPT_DIR)/flit.py $(INST_SHAREDIR)/scripts/
	install -m 0755 $(SCRIPT_DIR)/flit_*.py $(INST_SHAREDIR)/scripts/
	install -m 0644 $(SCRIPT_DIR)/flitutil.py $(INST_SHAREDIR)/scripts/
	install -m 0644 $(SCRIPT_DIR)/flitelf.py $(INST_SHAREDIR)/scripts/
	install -m 0644 $(SCRIPT_DIR)/README.md $(INST_SHAREDIR)/scripts/
	install -m 0644 $(DOC_DIR)/*.md $(INST_SHAREDIR)/doc/
	install -m 0644 $(DATA_DIR)/Makefile.in $(INST_SHAREDIR)/data/
	install -m 0644 $(DATA_DIR)/Makefile_bisect_binary.in $(INST_SHAREDIR)/data/
	install -m 0644 $(DATA_DIR)/custom.mk $(INST_SHAREDIR)/data/
	install -m 0644 $(DATA_DIR)/main.cpp $(INST_SHAREDIR)/data/
	install -m 0644 $(DATA_DIR)/tests/Empty.cpp $(INST_SHAREDIR)/data/tests/
	install -m 0644 $(DATA_DIR)/db/tables-sqlite.sql $(INST_SHAREDIR)/data/db/
	install -m 0644 $(CONFIG_DIR)/version.txt $(INST_SHAREDIR)/config/
	install -m 0644 $(CONFIG_DIR)/flit-default.toml.in $(INST_SHAREDIR)/config/
	install -m 0644 $(LITMUS_TESTS) $(INST_SHAREDIR)/litmus-tests/
	install -m 0644 LICENSE $(INST_LICENSEDIR)
	cp -r benchmarks/* $(INST_SHAREDIR)/benchmarks/
	@$(call color_out,CYAN,  Generating $(INST_FLIT_CONFIG))
	@# Make the flitconfig.py script specifying this installation information
	@echo "'''"                                                                  > $(INST_FLIT_CONFIG)
	@echo "Contains paths and other configurations for the flit installation."  >> $(INST_FLIT_CONFIG)
	@echo "This particular file was autogenerated at the time of installation." >> $(INST_FLIT_CONFIG)
	@echo "This is the file that allows installations to work from any prefix." >> $(INST_FLIT_CONFIG)
	@echo "'''"                                                                 >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "import os"                                                           >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "all = ["                                                             >> $(INST_FLIT_CONFIG)
	@echo "    'version',"                                                      >> $(INST_FLIT_CONFIG)
	@echo "    'script_dir',"                                                   >> $(INST_FLIT_CONFIG)
	@echo "    'doc_dir',"                                                      >> $(INST_FLIT_CONFIG)
	@echo "    'lib_dir',"                                                      >> $(INST_FLIT_CONFIG)
	@echo "    'include_dir',"                                                  >> $(INST_FLIT_CONFIG)
	@echo "    'config_dir',"                                                   >> $(INST_FLIT_CONFIG)
	@echo "    'data_dir',"                                                     >> $(INST_FLIT_CONFIG)
	@echo "    'litmus_test_dir',"                                              >> $(INST_FLIT_CONFIG)
	@echo "    ]"                                                               >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "_scriptpath = os.path.dirname(os.path.abspath(__file__))"            >> $(INST_FLIT_CONFIG)
	@echo "_prefix = os.path.realpath("                                         >> $(INST_FLIT_CONFIG)
	@echo "    os.path.join(_scriptpath, '..', '..', '..'))"                    >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# flit scripts"                                                      >> $(INST_FLIT_CONFIG)
	@echo "script_dir = os.path.join(_prefix, 'share', 'flit', 'scripts')"      >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# flit documentation"                                                >> $(INST_FLIT_CONFIG)
	@echo "doc_dir = os.path.join(_prefix, 'share', 'flit', 'doc')"             >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# compiled libflit.so"                                               >> $(INST_FLIT_CONFIG)
	@echo "lib_dir = os.path.join(_prefix, 'lib')"                              >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# flit C++ include files, primarily flit.h"                          >> $(INST_FLIT_CONFIG)
	@echo "include_dir = os.path.join(_prefix, 'include', 'flit')"              >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# default configuration for flit init"                               >> $(INST_FLIT_CONFIG)
	@echo "config_dir = os.path.join(_prefix, 'share', 'flit', 'config')"       >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# current version"                                                   >> $(INST_FLIT_CONFIG)
	@echo "version_file = os.path.join(config_dir, 'version.txt')"              >> $(INST_FLIT_CONFIG)
	@echo "with open(version_file, 'r') as _version_file_opened:"               >> $(INST_FLIT_CONFIG)
	@echo "    version = _version_file_opened.read().strip()"                   >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# default data files such as Makefile.in and main.cpp"               >> $(INST_FLIT_CONFIG)
	@echo "data_dir = os.path.join(_prefix, 'share', 'flit', 'data')"           >> $(INST_FLIT_CONFIG)
	@echo                                                                       >> $(INST_FLIT_CONFIG)
	@echo "# directory containing litmus tests"                                 >> $(INST_FLIT_CONFIG)
	@echo "litmus_test_dir = os.path.join("                                     >> $(INST_FLIT_CONFIG)
	@echo "    _prefix, 'share', 'flit', 'litmus-tests')"                       >> $(INST_FLIT_CONFIG)

.PHONY: uninstall
uninstall:
	@$(call color_out,BLUE,Uninstalling...)
	$(RMDIR) $(INST_INCLUDEDIR)
	$(RMDIR) $(INST_SHAREDIR)
	$(RMDIR) $(INST_LICENSEDIR)
	$(RM) $(INST_BINDIR)/flit
	$(RM) $(INST_LIBDIR)/$(notdir $(TARGET))
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)/include
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)/share/licenses
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)/share
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)/bin
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)/lib
	-rmdir --ignore-fail-on-non-empty $(EFFECTIVE_PREFIX)
