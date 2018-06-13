# FLiT Installation

[Table of Contents](README.md)
|
[Next](litmus-tests.md)

Instruction Contents:

* [Roles](#roles)
* [Prerequisites](#prerequisites)
  * [Shared Prerequisites](#shared-prerequisites)
  * [Runner Prerequisites](#runner-prerequisites)
  * [Database Prerequisites](#database-prerequisites)
* [FLiT Setup](#flit-setup)
* [Database Setup](#database-setup)
* [SSH Keys (Optional)](#ssh-keys-optional)

## Roles

FLiT is designed to build and execute its test suite on a variety of hosts and
compilers.  There are 3 roles used in a FLiT architecture:

* **Launcher**: Where you start a large run which remotely executes on
  **Runner** boxes
* **Runner**: Compiles different versions of reproducibility tests and executes
  them to generate test results.  You can have more than one of these.
* **Database**: Stores the results for later analysis

Each role requires slightly different setup.  A particular computer may only
satisfy one role, or it may satisfy multiple roles.  You can potentially have
all three roles on a single machine, which is the simplest use case.

## Prerequisites

### Shared Prerequisites

The following prerequisites are required for all three roles:

* [git](https://git-scm.com)
* [bash](https://www.gnu.org/software/bash)
* [python3](https://www.python.org)
  * The [toml](https://github.com/uiri/toml) module (for
    [TOML](https://github.com/toml-lang/toml) configuration files)
* [make](https://www.gnu.org/software/make)
* [gcc](https://gcc.gnu.org) version 4.9 or higher

For Debian-based systems:

```bash
sudo apt install bash build-essential git python3 python3-toml
```

For homebrew on OSX (besides installing [Xcode](https://developer.apple.com/xcode))

```bash
brew install make python3 gcc git
pip3 install toml
```

If you install python version 3.0 or later, then you will need to have a
symbolic link called `python3` in your `PATH` pointing to that python executable.

### Runner Prerequisites

The test runner can run multiple compilers.  For now, only one compiler is
supported from each of the types: GCC, Clang, Intel's `icpc`, and NVIDIA's `nvcc`.
Simply have the one you want used as the first in your system PATH.  You do not
need all four of those, only those ones installed will be used.  But all of
them need to support C++11.

If this is not on the same machine as the Launcher, then the Database machine
will need an SSH server running.

```bash
sudo apt install openssh-server
```

### Database Prerequisites

Postgres used to be the supported database for FLiT.  That has since been
changed to using SQLite3.  Because of this, all you need is `sqlite3` installed,
which you should already have installed because `python3` already requires it.

## FLiT Setup

All three roles will need FLiT available and compiled.  It can be optionally
installed as well.

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
tool from the git repository, simply make a symbolic link to the flit.py
script.  Here I presume that `$HOME/bin` is in your `PATH` variable

```bash
ln -s ~/FLiT/scripts/flitcli/flit.py ~/bin/flit
```

See [FLiT Command-Line](flit-command-line.md) for more information on how to
use the command-line tool.

## Database Setup

There should be nothing to setup since `sqlite3` should already be present.

Creating sqlite3 databases is currently working for the **Runner** machines,
but the central **Database** system is not currently supported.  We plan to fix
this functionality and support a separate **Database** system.

## SSH Keys (Optional)

The automation for executing the reproducibility FLiT tests on multiple hosts
heavily relies on SSH.  It would make your life easier to have SSH keys paired
so that you do not need to enter a password when logging in remotely to
**Runner** machines.

If the roles are separated onto different machines, the automation will attempt
to make the following connections:

* **Launcher** -> **Runner**
* **Launcher** -> **Database**
* **Runner** -> **Database**

so these are the connections you may want to setup SSH keys for.  See
[Ubuntu's help
documentation](https://help.ubuntu.com/community/SSH/OpenSSH/Keys) for setting
up SSH keys.

[Table of Contents](README.md)
|
[Next](litmus-tests.md)
