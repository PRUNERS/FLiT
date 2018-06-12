# This file is included at the end of the copied Makefile.  If you have some
# things you want to change about the Makefile, it is best to do it here.

# additional source files to compile other than what is in '.' and 'tests/'
# since those directories are added by a wildcard.
SOURCE         +=

# required compiler flags
# for example, include directories
#   CC_REQUIRED += -I<path>
# or defines
#   CC_REQUIRED += -DDEBUG_ENABLED=1
CC_REQUIRED    +=

# required linker flags
# for example, link libraries
#   LD_REQUIRED += -L<library-path> -l<library-name>
# or rpath
#   LD_REQUIRED += -Wl,-rpath=<abs-path-to-library-dir>
LD_REQUIRED    +=

# compiler and linker flags respectively - specifically for a dev build
# - DEV_CFLAGS:   non-recorded compiler flags (such as includes)
# - DEV_LDFLAGS:  linker flags (also not under test)
DEV_CFLAGS     +=
DEV_LDFLAGS    +=

