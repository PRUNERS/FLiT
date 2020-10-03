# FLiT Installation

[Prev](release-notes.md)
|
[Table of Contents](README.md)
|
[Next](litmus-tests.md)

Instruction Contents:

* [Prerequisites](#prerequisites)
  * [Compilers](#compilers)
  * [Optional Dependencies](#optional-dependencies)
* [FLiT Setup](#flit-setup)
* [Bash Completion](#bash-completion)
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
* [make](https://www.gnu.org/software/make)
* A C++11 compatible compiler
  (see section [Compilers](#compilers) for supported versions)
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
sudo apt install python3-toml
```

or with `pip`

```bash
sudo apt install python3-pip
pip3 install --user toml
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

FLiT officially supports the use of GCC, Clang, and Intel compilers for C++.
You may try to use other compilers, but they are not yet officially supported.

Supported compiler versions:

| Compiler Type | Minimum Supported Version |
|:-------------:|:-------------------------:|
| gcc           | 4.9.0                     |
| clang         | 3.4.0                     |
| intel         | 16.0                      |

If your compiler version is less than the supported versions and you want FLiT to
support it, please create an [issue](https://github.com/PRUNERS/FLiT/issues)
and we may be able to add support for you.  Otherwise, you are on your own.

Likewise, if you want support added for other types of compilers, such as the
PGI compiler or the IBM compiler, please create an
[issue](https://github.com/PRUNERS/FLiT/issues).


### Compiling FLiT

FLiT no longer needs to be compiled.  A design decision was made to have each
test executable compilation to compile in the FLiT source code directly.  This
decision was made specifically because of the C++ standard library
incompatibility (see [Standard C++ Library
Implementations](standard-c++-library-implementations.md)).  Before, when FLiT
was compiled as `libflit.so`, it restricted the user to only the standard
library used to compile that library.

Now that FLiT is compiled into the executable, the user may play around with
many different standard library implementations.  It makes the installation of
FLiT much more flexible as well.

The downside is the recompilation of the FLiT source code over and over again.
To partially alleviate this concern, we used an approach of a [Unity
Build](https://buffered.io/posts/the-magic-of-unity-builds/).  Please refer to
[Issue #210](https://github.com/PRUNERS/FLiT/issues/210) to see the discussion
behind this design decision, along with the pros and cons.


### Optional Dependencies

The FLiT source code is compiled dynamically with your FLiT tests.  Therefore,
it only matters what optional dependencies are installed at the time of using
your FLiT tests, not at the time of installation.

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

FLiT also officially supports Intel's MPI implementation.  We would like to
support all implementations of MPI, but on an as-needed basis.  If you are
using a different implementation of MPI and you find it does not work with
FLiT, please submit an [issue](https://github.com/PRUNERS/FLiT/issues).

## FLiT Setup

```bash
git clone https://github.com/PRUNERS/FLiT.git
cd FLiT
```

Since FLiT is not pre-compiled, you may be wondering what the top-level
`Makefile` is for.  The default target is the `help` target explaining about
the available targets.

```bash
FLiT is an automation and analysis tool for reproducibility of
floating-point algorithms with respect to compilers, architectures,
and compiler flags.

The following targets are available:

  help        Shows this help message and exits (default)
  install     Installs FLiT.  You may override the PREFIX variable
              to install to a different directory.  The default
              PREFIX value is /usr.
                exe: "make install PREFIX=$HOME/installs/usr"
  uninstall   Delete a FLiT installation.  You must use the same
              PREFIX value used during installation.
  check       Run tests for FLiT framework
  clean       Clean up after the tests
```

You do not need to install FLiT in order to use it.  It may be either used from
the Git repository (or extracted release tar file), or you may install with
whatever PREFIX you prefer.

To use FLiT from the repository, simply use the `scripts/flitcli/flit.py`
script directly.  You may use an alias for it or a symbolic link.

If you choose to install

```bash
make install
```

The `flit` command-line tool will be symbolically linked as `$PREFIX/bin/flit`

_Note: if you want to see all of the installation details, set `VERBOSE=1`,
a.k.a., `make install VERBOSE=1`_

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
it will work, granted you maintain the same directory hierarchy.  If it does
not work in this situation, then it is a bug -- please submit an
[issue](https://github.com/PRUNERS/FLiT/issues).

If you do not want to install somewhere and want to use the flit command-line
tool from the cloned git repository, simply make a symbolic link to the
`flit.py` script.  Here I presume that `$HOME/bin` is in your `PATH` variable

```bash
ln -s ~/FLiT/scripts/flitcli/flit.py ~/bin/flit
```

See [FLiT Command-Line](flit-command-line.md) for more information on how to
use the command-line tool.

## Bash Completion

FLiT comes with a bash-completion script to be used with bash.  In the
repository, the script can be found at `scripts/bash-completion/flit`.  If you
install FLiT using `make install`, it gets installed to
`$PREFIX/share/bash-completion/completions/flit`.  Some systems will be able to
use the flit bash completion from this location directly (if `$PREFIX="/usr"`).
Other systems store these scripts in `/etc/bash_completion.d/`.

If your system stores bash-completion scripts in `/etc/bash_completion.d/`, you
can either copy the script, or create a symbolic link (preferred).

```bash
sudo ln -s /usr/share/bash-completion/completions/flit /etc/bash_completion.d/flit
```

If you do not have sudo permissions or do not want to install bash-completion
for flit system-wide, then you can implement it locally for your user account.
Newer bash-completion installations allow users to have a script in their home
directory called `$HOME/.bash_completion`.  We recommend you have a directory
for storing bash-completion scripts.  You can put the following in your
`$HOME/.bash_completion` file

```bash
if [ -d ~/.local/share/bash-completion/completions ]; then
  for file in ~/.local/share/bash-completion/completions/*; do
    if [ -f "$file" ]; then
      source "$file"
    fi
  done
fi
```

Then you can simply copy or symbolically link bash-completion scripts into
`~/.local/share/bash-completion/completions`.  If you are using FLiT from the
repository, you can accomplish that with

```bash
mkdir -p ~/.local/share/bash-completion/completions
ln -s <git-repo-dir>/FLiT/scripts/bash-completion/flit ~/.local/share/bash-completion/completions/flit
```

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
