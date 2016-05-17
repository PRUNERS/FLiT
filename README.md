# QFP #
A project to quickly detect discrepancies in floating point computation across hardware, compilers, libraries and software.

This project has taken on two or three approaches (branches).  Here, however, we will discuss the 'litmus test' version and detail how to get it set up.

QFP consists of two parts: QC, or 'quick classifier'; and QD, 'quick differential
debugger.

QFP-LT (litmus test) works by running the test set on a cross product
set of compilers, flags, precisions and hosts, where each test generates
a 'score'.  This score should be an approximate error value, where 0 would
be no error.  This score is used to group tests (where a test is a {t,c,f,p,h} for
**t**est, **c**ompiler, **f**lags, **p**recision, **s**ort, and **h**ost)
into equivalence classes.  Test results are stored in a PostgreSQL relational
database.  This is the QC part of QFP.

Pairs of tests can be chosen to run in the differential debugger (QD).
Of course, the test pairs should be based on the same litmus test (t),
and currently, are limited to running on the same host.

## Prerequisites and Setup ##

QFP-LT is designed to build and excute its test suite on a variety of
hosts and compilers.  However, there are two types of hosts whose
environments must be considered -- the _primary_ host, from which you
will be envoking the tests, and zero or more  _remote_ hosts, where
tests may also execute and return data to the _primary_.

### Software ###

These are required on the primary host; entries demarcated with
a *** are required on the _remote_ hosts.

* python3*
* gcc 5.2+*
* git (used from 1.7.1 to 2.5)*
* gdb 7.11 
** The bleeding-edge is a little unstable (esp regarding the python API interface),
   but this version works (and earlier
   ones do not)
* PostgreSQL 9.4.7
** not sure how version variance affects things such as importing QC database

On Ubuntu 15.10, everything was obtained with apt-get satisfactorily,
except we have to build gdb from binutils-gdb.  Hopefully the gdb-python
API will stabilize in the next release.

#### Building binutils-gdb ####

This describes a global installation.  You may use **--prefix=[path]**
on the  configure command line to install to a different root
(such as /home/[your_login]/software). Of course, you'll want to adjust
your $PATH accordingly (i.e. cat 'export PATH=$PATH:/home/fred/software

Here are the steps:
{} = optional; [] = fill in the blank
```
cd [build location]
wget https://github.com/bminor/binutils-gdb/archive/gdb-7.11-release.tar.gz
tar xf gdb-7.11-release.tar.gz
cd binutils-gdb-gdb-7.11-release
mkdir build && cd build
../configure --with-python=$(which python3) {--prefix=/home/you/mysoftware}
make -j [num procs]
{sudo} make install
```

### Configuring test (QC) run ###

This is where the QC run is configured.

There are a few things you can control regarding the parameters
that will be covered in the tests.

As previously mentioned, the parameters are **t**est, **c**ompiler, **f**lags, **p**recision, and **h**ost.

This table explains the parameters.  You are able to add tests, compilers, hosts and compiler flags.
The table below explains how this is accomplished.

| parameter | file | example | notes |
---|---|---|---
compiler | perpTest/Makefile2 | COMPILERS := $(foreach c, GCC | Add a var for your compiler and add to COMPILERS line
compiler flag | Makefile.switches | FPMODFST2 := -fp-model fast=2 | [unique name] := [flag] <br> also add name to **SWITCHES** list declaration in file
host name | collectAll.py | youruname@kingspeak2.chpc.utah.edu | username@FQDN of host, added to hostinfo var (at top of file)
test name |  | perpTest/test.cpp | There are three steps to adding your own test.  See below


#### adding a new test ####

As an example, examine the function _DoSkewSymCPRotationTest_ in perpVects/test.cpp.
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


#### Database configuration ####

After you have set up your PosgreSQL database, you may initialize the database (add the required tables)
with these commands:

```
cd db/db_backups
psql qfp < dumpfile
cd ../psql_commands
psql qfp < create_importQFPResults
```

## Running QFPC (or QC -- the classifier) ##

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

More sample queries are available at: https://github.com/Geof23/QFP/wiki/Some-Analysis


## QD ##

To execute QD, you should take the following steps:

* Log onto a host that you'd like to examine a pair of tests (for instance, kingspeak2.chpc.utah.edu:/remote_qfp)
[for example]:
* cd qfp/perpVects
* make -f MakefileQD -j 12
* runQD [test1] [test2] [precision1] [sort1] [precision2] [sort2]

