# FLiT #

[![FLiT Bird](/flit-small.png)](https://github.com/PRUNERS/FLiT "FLiT")

Floating-point Litmus Tests is a test infrastructure for detecting varibility
in floating-point code caused by variations in compiler code generation,
hardware and execution environments.

FLiT works by building many versions of the test suite, using multiple c++
compilers, floating-point related settings (i.e. flags) and optimization
levels.  These tests are then executed on target platforms, where a
representative 'score' is collected into a database, along with the other
parameters relevant to the execution, such as host, compiler configuration and
compiler vendor.  In addition to the user-configured test output, we collect
counts of each assembly opcode executed (currently, this works with Intel
architectures only, using their PIN dynamic binary instrumentation tool).

After executing the suite and collecting the data, it is easy to see how
results may diverge using only different compiler settings, etc.  Also, the
developer is able to understand how to configure their build environment for
their target architecture(s) such that they can expect consistent
floating-point computations.

It consists of the following components:

* a test infrastructure in the form of c++ code, where additional tests
  are easily added
* a dynamic make system to generate diverse executables
* an execution disbursement system
* a SQL database for collecting results
  * a collection of queries to help the user understand results
  * some data analysis tools, utilizing machine intelligence (such as k-means
  clustering)

Contents:

* [Prerequisites and Setup](#prerequisites-and-setup)
  * [Clone this repository](#clone-and-configure-flit-git)
  * [Software](#software)
    * [python3, including the _dev_ package](#install-python3)
    * [gcc 5.2+](#install-gcc)
    * [PostgreSQL 9.4.7+](#configuring-postgresql-database)
    * [Intel PIN 3.0+](#download-pin)
    * [CUDA Toolkit 7.5+](#setup-cuda-7.5)
  * [Configuring FLiT](#configuring-test-flit-run)
  * [Adding a new test](#adding-a-new-test)
  * [Running FLiT](#running-qc)
  * [Examining data &mdash; a sample query](#sample-query)


## Prerequisites and Setup ##

FLiT is designed to build and excute its test suite on a variety of hosts and
compilers.  However, there are three types of hosts whose environments must be
considered &mdash; the _primary_ host, from which you will be envoking the
tests, a _database_ host, and zero or more  _remote_ hosts, where tests may also execute and
return data to the _primary_.

Also, these instructions assume mostly that you have *root* access on your
system.  It is possible with many package managers and source builds to install
locally, but is beyond the scope of these instructions.

Our demonstration system for these instructions is Ubuntu &mdash; other Unix
based systems should work similarly (including Mac OSX 10).

### Clone FLiT ###

In order to clone FLiT, you will need git on your system.  If you do not have
git, then install it:

```sudo apt-get install git```

After you have installed git, you can clone this repository

```
cd [location for FLiT to live]
git clone https://github.com/Geof23/QFP.git
cd QFP
git checkout unified_script
```

### Software ###

Here we'll describe the dependencies that you must have on your systems to use
FLiT.  There are again, two types of systems: the primary system that you will
directly execute commands from and the remote systems that will collect data.

* [python3, python3-dev](#install-python3)
* [gcc 5.2+](#install-gcc)
* [git (used from 1.7.1 to 2.5)](#install-git)
* [PostgreSQL 9.4.7+](#configuring-postgresql-database)
* [FLiT](#clone-and-configure-flit-git)

. . . and the list for remote/secondary hosts:

* [python3](#install-python3)
* [gcc 4.9+](#install-gcc)
* [git (used from 1.7.1 to 2.5)](#install-git)
* [Intel PIN 3.0+](download-pin)

. . . and optionally (for CUDA):

* [CUDA Toolkit 7.5+](#setup-cuda)


On Ubuntu 15.10, everything was obtained with apt-get satisfactorily, except we
have to build gdb from the binutils-gdb (unless your package manager provides
at least 7.11).

#### Install python3 ####

Many systems will already provide _python3_.

```sudo apt-get install python3 python3-dev```

#### Install gcc ####

Again, recent systems already have a new enough version.
The required version isn't as high for remote hosts,
but this is a minimum.

```sudo apt-get install gcc-[5.2|4.9]```

Another note: _CUDA 7.5_ requires gcc 4.9, so when installing
gcc on the _remote_ hosts, you must have this version.  It is
recommended that you install _both_ versions.  FLiT will
use 4.9 for CUDA and 5.2 for the rest of the test suite.

#### Setup CUDA 7.5 ####

This is beyond the scope of this document, but the NVidia instructional
documents are quite helpful.  This is step is _optional_, only required
if you'd like to explore the variability of CUDA kernels.

#### Configuring PostgreSQL Database ####

To take advantage of the query and analysis capabilities of a SQL database
consisting of the QC (classifier) data, you will need to install PostgreSQL.

Here are instructions that work on a Debian (apt-get) package management
system.  Other approaches are possible such as using other package managers,
or building from source.

##### Install PostgreSql #####

Note, you will need equivalent to the following packages, be it through
another package system or building from source (later versions are OK).

```
sudo apt-get install postgresql-9.4 postgresql-plpython3-9.4
```
After this step, you should have a running DB server, and a new user
account on your system, _postgres_.

Try `ps aux | grep postgres` to see if your DB is running.  You will see
something similar to:

```
you@somemachine$ ps aux | grep postgres
postgres  3062  0.0  0.2 288160 24196 ?        S    May17   0:00 /usr/lib/postgresql/9.4/bin/postgres -D /var/lib/postgresql/9.4/main -c config_file=/etc/postgresql/9.4/main/postgresql.conf
postgres  3064  0.0  0.1 288296 10580 ?        Ss   May17   0:00 postgres: checkpointer process
postgres  3065  0.0  0.0 288160  5612 ?        Ss   May17   0:00 postgres: writer process
postgres  3066  0.0  0.0 288160  8620 ?        Ss   May17   0:00 postgres: wal writer process
postgres  3067  0.0  0.0 288584  6428 ?        Ss   May17   0:00 postgres: autovacuum launcher process
postgres  3068  0.0  0.0 143416  4276 ?        Ss   May17   0:00 postgres: stats collector process
you   15850  0.0  0.0   8216  2164 pts/8    R+   17:18   0:00 grep --color=auto postgres
```

Also, this output shows that the _postgres_ user has been added to the system.

We include a SQL script that you may use to easily configure
the FLiT database.

'''
cd db
./create_database.sh
'''

This will create the flit database and add the required schemas.


##### Configure FLiT #####

There are a few things you can control regarding the parameters
that will be covered in the tests.

As previously mentioned, the parameters are **t**est, **c**ompiler, **f**lags, **p**recision, and **h**ost.

This table explains the parameters.  You are able to add tests, compilers, hosts and compiler flags.
The table below explains how this is accomplished.

| parameter | file | example | notes |
---|---|---|---
compiler | perpTest/Makefile | COMPILERS := $(foreach c, GCC | Add a var for your compiler and add to COMPILERS line
compiler flag | Makefile.switches | FPMODFST2 := -fp-model fast=2 | [unique name] := [flag] <br> also add name to **SWITCHES** list declaration in file
host name | collectAll.py | youruname@kingspeak2.chpc.utah.edu | username@FQDN of host, added to hostinfo var (at top of file)
test name |  | perpTest/test.cpp | There are three steps to adding your own test.  See below


#### adding a new test ####

As an example, examine the function _DoSkewSymCPRotationTest_ in qfpc/test.cpp.
You need the following features:

* return FPTests::resultType
* should be a **template<typename T>**, allowing for multiple precision (T is the main float type used in your computation)
* you should accumulate results, or have another key FP type that is used in your computation, be a global value
* add a line in this form:

```
#ifndef QC
    QFP::checkpoint(Globals<T>::sum, sizeof(Globals<T>::sum), "sum", NO_WATCH);
    #endif

```

Where the parameters to _checkpoint_ are:

1 global variable reference
1 size of variable
1 a name tag for variable
1 boolean, whether to ignore watch (for QC)



## Running QC ##

(the QFPC Classifier)

execute _collectAll.py_

This script will push the project out to the hosts that you indicate in this datastructure (as noted above):

```
hostinfo = [['u0422778@kingspeak.chpc.utah.edu', 12],
            ['sawaya@bihexal.cs.utah.edu', 24],
            ['sawaya@gaussr.cs.utah.edu', 4],
            ['sawaya@ms0123.utah.cloudlab.us', 1]]
```

For each host, the _hostCollect.py_ script is executed, running the makefile, which
builds and runs the tests for all configurations.  Results are groomed and collected
in the results directory (on the remote), which is then pulled back to the master host
(where _collectAll.py_ was executed).

After all results are collected, this script then runs the stored function in the database,
_importQFPResults_, on the data collected from all hosts.  It is then available to query.


### sample query ###

You may log into the PostgreSQL client, and examine your data:

```
qsql qfp
select * from tests;
```

More sample queries are available [at the wiki](https://github.com/Geof23/QFP/wiki/Some-Analysis).


## QD ##

To execute QD, you should take the following steps:

* Log onto a host that you'd like to examine a pair of tests (for instance, kingspeak2.chpc.utah.edu:/remote_qfp)
[for example]:
* cd qfp/qfpc
* make -f MakefileQD -j 12
* runQD [test1] [test2] [precision1] [sort1] [precision2] [sort2]
