# FLiT Installation

[Table of Contents](README.md)
|
[Next](litmus-tests.md)

Instruction Contents:

* [Prerequisites](#prerequisites)
  * [Compilers](#compilers)
  * [Optional Dependencies](#optional-dependencies)
* [FLiT Setup](#flit-setup)
* [Database Setup](#database-setup)
* [Uninstallation](#uninstallation)

## Prerequisites

Stuff you should already have:

* [bash](https://www.gnu.org/software/bash)
* [binutils](https://www.gnu.org/software/binutils)
* [coreutils](https://www.gnu.org/software/coreutils/coreutils.html)
* hostname

Stuff you may need to get

* [git](https://git-scm.com)
* [python3](https://www.python.org)
  * The [toml](https://github.com/uiri/toml) module (for
    [TOML](https://github.com/toml-lang/toml) configuration files)
* [make](https://www.gnu.org/software/make)
* [gcc](https://gcc.gnu.org) version 4.9 or higher
* [sqlite3](https://sqlite.org) version 3.0 or higher.
  You can use the one that comes with python, or install as a standalone.

For Debian-based systems:

```bash
sudo apt install \
  bash binutils build-essential coreutils git hostname \
  python3 python3-toml
```

For homebrew on OSX (besides installing [Xcode](https://developer.apple.com/xcode))

```bash
brew install make python3 gcc git
pip3 install toml
```

If you install python version 3.0 or later, then you will need to have a
symbolic link called `python3` in your `PATH` pointing to that python
executable.

### Compilers

The GCC compiler is the only one required for installation of FLiT since it is
used to compiler the FLiT shared library.  Other than that, you are free to
install another version of GCC, as well as Clang and the Intel compiler.  If
you are missing either Clang or the Intel compiler, FLiT will still work as
expected.

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

## FLiT Setup

You will need FLiT available and compiled.  It can be optionally installed.

```bash
git clone https://github.com/PRUNERS/FLiT.git
cd FLiT
make
```

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
`PREFIX` value, then that custom `PREFIX` value should also be used.  For
example if you installed with

```bash
make install PREFIX=~/my-installs/
```

then you would uninstall with

```bash
make uninstall PREFIX=~/my-installs/
```

[Table of Contents](README.md)
|
[Next](litmus-tests.md)
