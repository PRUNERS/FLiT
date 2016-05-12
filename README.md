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

## Configuring test (QC) run ##

This is where the QC run is configured.

There are a few things you can control regarding the parameters
that will be covered in the tests.

As previously mentioned, the parameters are **t**est, **c**ompiler, **f**lags, **p**recision, and **h**ost.

This table explains the parameters.  Some can be specified by a file, others
will be explored in full (i.e. {configurable set} x {precision} x {sort}).
Configurable parameter files are prepopulated, and you may comment a line with a '#' in
the first column.

| parameter | file | example | notes |
---|---|---|---
compiler | compilers | clang-3.6 | compiler path
compiler flag | Makefile.switches | FPMODFST2 := -fp-model fast=2 | [unique name] := [flag] <br> also add name to **SWITCHES** list declaration in file
host name | hosts | youruname@kingspeak2.chpc.utah.edu | username@FQDN of host
test name | tests | fpTestSuite::MyTest | the namespace qualified name of your test.<br> You should
subclass fpTestSuite::TestSuper
