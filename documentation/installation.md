# FLiT Installation

[Prev](release-notes.md)
|
[Table of Contents](README.md)
|
[Next](litmus-tests.md)

Instruction Contents:

* [Prerequisites](#prerequisites)
  * [Compilers](#compilers)
  * [Clang Only](#clang-only)
  * [Non-Standard GCC Installs](#non-standard-gcc-installs)
  * [Optional Dependencies](#optional-dependencies)
* [FLiT Setup](#flit-setup)
* [Database Setup](#database-setup)
* [Uninstallation](#uninstallation)

## Prerequisites

Stuff you should already have:

* [bash](https://www.gnu.org/software/bash)
* [binutils](https://www.gnu.org/software/binutils) version 2.26 or higher
* [coreutils](https://www.gnu.org/software/coreutils/coreutils.html)
* hostname

Stuff you may need to get

* [git](https://git-scm.com)
* [python3](https://www.python.org)
  * [toml](https://github.com/uiri/toml) module (for
    [TOML](https://github.com/toml-lang/toml) configuration files)
  * (optional) [pyelftools](https://github.com/eliben/pyelftools) module for
    parsing ELF files.  This is used for `flit bisect`; all other functionality
    will work without it.
* [make](https://www.gnu.org/software/make)
* [gcc](https://gcc.gnu.org) version 4.9 or higher (or
  [clang](https://clang.llvm.org), see section [Clang Only](#clang-only))
* [sqlite3](https://sqlite.org) version 3.0 or higher.
  You can use the one that comes with python, or install as a standalone.

For Debian-based systems:

```bash
sudo apt install \
  bash binutils build-essential coreutils git hostname \
  python3
```

The python modules can be installed with `apt`

```bash
sudo apt install python3-toml python3-pyelftools
```

or with `pip`

```bash
sudo apt install python3-pip
pip3 install --user toml pyelftools
```

For homebrew on OSX (besides installing [Xcode](https://developer.apple.com/xcode))

```bash
brew install make python3 gcc git
pip3 install toml pyelftools
```

If you install python version 3.0 or later, then you will need to have a
symbolic link called `python3` in your `PATH` pointing to that python
executable.


### Compilers

The GCC compiler is the only one required for installation of FLiT since it is
used to compile the FLiT shared library.  Other than that, you are free to
install another version of GCC, as well as Clang and the Intel compiler.  If
you are missing either Clang or the Intel compiler, FLiT will still work as
expected.

The supported compiler versions are:

| Compiler Type | Minimum Supported Version |
|:-------------:|:-------------------------:|
| gcc           | 4.9.0                     |
| clang         | 3.4.0                     |
| intel         | 16.0                      |

If your compiler version is below those on this list and you want FLiT to
support it, please create an [issue](https://github.com/PRUNERS/FLiT/issues)
and we may be able to add support for you.  Otherwise, you are on your own.

Likewise, if you want support added for other types of compilers, such as the
PGI compiler or the IBM compiler, please create an
[issue](https://github.com/PRUNERS/FLiT/issues).


### Clang Only

FLiT is mostly geared around having at least GCC around, however, users may
want to skip using GCC and use Clang instead.  If this is your use case, this
can be done.

To compile FLiT using Clang, set the `CXX` environment variable to the
executable for Clang you wish to use.  For example:

```bash
git clone https://github.com/PRUNERS/FLiT.git
cd FLiT
export CXX=clang
make
sudo make install
```

Then when creating your environment, simply provide only a Clang compiler.
This setup is largely untested, so if you have trouble, please submit an
[issue](https://github.com/PRUNERS/FLiT/issues).


### Non-Standard GCC Installs

Warning: the FLiT shared library should be compiled with the same version of
GCC that will be used by the FLiT tests.  If you do not adhere to this, you may
get linker errors because of using a different implementation of the C++
standard library (i.e. libstdc++.so).  FLiT uses C++11, so the compiler used
should be able to compile C++11 code.

If the GCC compiler you want to use is not in the standard installation
location, you can still use it.  As an example, suppose I do not want to use
the GCC installed on the system, but instead I want to use the one I compiled
myself and installed into `$HOME/installs/gcc-5.5.0`.  The way I would compile
the FLiT shared library would be:

```bash
CXX="$HOME/installs/gcc-5.5.0/bin/g++" LDFLAGS="-Wl,-rpath=$HOME/installs/gcc-5.5.0/lib64" make
```

The `CXX` flag could be specified after `make`, but `LDFLAGS` should be
specified before it so that it prepends the variable rather than completely
overriding it.  If you were to have done the following instead:

```bash
make CXX="$HOME/installs/gcc-5.5.0/bin/g++" LDFLAGS="-Wl,-rpath=$HOME/installs/gcc-5.5.0/lib64"
```

It would delete the other necessary `LDFLAGS` set within the `Makefile`.  This
would be the approach if you really desired to override the internal settings
of the variable.

Note that without the `LDFLAGS` variable, the wrong version of `libstdc++.so`
would be used at runtime, as it would likely find the one from the installed
GCC rather than the one I compiled myself.  Another solution instead of using
rpath is to set the `LD_LIBRARY_PATH` before running anything with FLiT, but it
is recommended to use rpath instead.


### Optional Dependencies

FLiT has [MPI support](mpi-support.md) which you may want to use.  To compile
and install FLiT, MPI does not need to be installed.  If you later choose the
use MPI, you only need it installed when you go to compile the tests that
require it.

If you choose to use MPI support, you likely know what you need.  FLiT requires
that both `mpic++` and `mpirun` are found in the system `PATH`.  On Ubuntu,
OpenMPI is installed with

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

Or you can alternatively use MPICH

```bash
sudo apt install mpich
```

## FLiT Setup

You will need FLiT available and compiled.  It can be optionally installed.

```bash
git clone https://github.com/PRUNERS/FLiT.git
cd FLiT
make
```

_Note: if you want to see all of the compilation details, set `VERBOSE=1`,
e.g., `make VERBOSE=1`_

You can either use the flit command-line tool from the git repository or you
can install it.  If you choose to install it, it is simply

```bash
sudo make install
```

If you want to specify an install prefix, you do that by specifying the
`PREFIX` variable to make

```bash
make install PREFIX=~/my-installs/
```

Additionally, there is a `DESTDIR` variable for specifying a path where the
installation will consider to be the root of the filesystem.  The given `PREFIX`
will be relative to the `DESTDIR` (if given).  See the
[GNU coding standard](https://www.gnu.org/prep/standards/html_node/DESTDIR.html)
for more information about `DESTDIR`.

Note, installations in `DESTDIR` are typically not guaranteed to work as-is
until they are copied to the given `PREFIX`, however FLiT guarantees that it
will work in both situations.  You may copy the files in `PREFIX` anywhere and
it will work, granted you maintain the same directory heirarchy.

If you do not want to install somewhere and want to use the flit command-line
tool from the cloned git repository, simply make a symbolic link to the
`flit.py` script.  Here I presume that `$HOME/bin` is in your `PATH` variable

```bash
ln -s ~/FLiT/scripts/flitcli/flit.py ~/bin/flit
```

See [FLiT Command-Line](flit-command-line.md) for more information on how to
use the command-line tool.

## Database Setup

There should be nothing to setup since `sqlite3` should already be present.

## Uninstallation

You can also uninstall as easily as you installed.  If you used a custom
`PREFIX` value and/or `DESTDIR` value, then simply provide those values again.
For example if you installed with

```bash
make install DESTDIR=~/my-installs PREFIX=/usr
```

then you would uninstall with

```bash
make uninstall DESTDIR=~/my-installs PREFIX=/usr
```

[Prev](release-notes.md)
|
[Table of Contents](README.md)
|
[Next](litmus-tests.md)
