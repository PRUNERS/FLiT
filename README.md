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

Contents:

* [Prerequisites and Setup](#prerequisites-and-setup)
  ** [Software](#software)
    *** [python3](#install-python3)
    *** [gcc 5.2+](#install-gcc)
    *** [git (used from 1.7.1 to 2.5)](#install-git)
    *** [gdb 7.11 ](#install-gdb)
    *** [PostgreSQL 9.4.7+](#configuring-postgresql-database)
    *** [QFP](#clone-and-configure-qfp-git)
  ** [Configuring QC](#configuring-test-(qc)-run)
    *** [Adding a new test](#adding-a-new-test)
* [Running QC (QFPC Classifier)](#running-qc)
  ** [Examining data &mdash; a sample query](#sample-query)
* [QD &mdash; the Differential Debugger](qd)


## Prerequisites and Setup ##

QFP-LT is designed to build and excute its test suite on a variety of
hosts and compilers.  However, there are two types of hosts whose
environments must be considered &mdash; the _primary_ host, from which you
will be envoking the tests, and zero or more  _remote_ hosts, where
tests may also execute and return data to the _primary_.

Also, these instructions assume mostly that you have *root* access
on your system.  It is possible with many package managers and
source builds to install locally, but is beyond the scope of
these instructions.

Our demonstration system for these instructions is Ubuntu &mdash;
other Unix based systems should work similarly (including Mac OSX 10).

### Software ###

Here we'll describe the dependencies that you must have on your
systems to use QFP.  There are again, two types of systems:
the primary system that you will directly execute commands
from and the remote systems that will collect data.

Most of these tools are very recent versions and you may have
challenges using QFP on an older system, or you will have to
build from source a number of tools and libraries.

We won't go into detail how to build anything but binutils-gdb, but
show the _apt-get_ tool's use for obtaining our dependencies.

Here is the software required on the primary host:

* [python3](#install-python3)
* [gcc 5.2+](#install-gcc)
* [git (used from 1.7.1 to 2.5)](#install-git)
* [gdb 7.11 ](#install-gdb)
  ** This is required, and at this writing, is the latest stable release.
* [PostgreSQL 9.4.7+](#configuring-postgresql-database)
* [QFP](#clone-and-configure-qfp-git)

. . . and the list for remote/secondary hosts:

* [python3](#install-python3)
* [gcc 4.9+](#install-gcc)
* [git (used from 1.7.1 to 2.5)](#install-git)


On Ubuntu 15.10, everything was obtained with apt-get satisfactorily,
except we have to build gdb from the binutils-gdb (unless your package
manager provides at least 7.11).

#### Install python3 ####

Many systems will already provide _python3_.

```sudo apt-get install python3```

#### Install gcc ####

Again, recent systems already have a new enough version.
The required version isn't as high for remote hosts,
but this is a minimum.

```sudo apt-get install gcc-[5.2|4.9]```

#### Clone and Configure QFP git ####

```
cd [location for QFP to live]
git clone https://github.com/Geof23/QFP.git
git branch rel_lt
```

#### Install git ####

It is hard to find a system without git.  But:

```sudo apt-get install git```

#### Configuring PosgreSQL Database ####

To take advantage of the query and analysis capabilities of a SQL database
consisting of the QC (classifier) data, you will need to install PostgreSQL.

Here are instructions that work on a Debian (apt-get) package management
system.  Other approaches are possible such as using other package managers,
or building from source.

##### Install PostgreSql #####

Note, you will need equivalent to the following packages, be it through
another package system or building from source.

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

##### Configure Database #####

The next step is to add your user name to the database, and create the _qfp_ database,
owned by you, and install python support:


```
sudo su postgres
psql
create user [your system username];
create database qfp owner [your system username];
\q
exit
psql qfp
create language plpython3u;
\q
```

##### Modify the restoration file #####

This is necessary to make you the owner of the tables, etc.

```
cd qfp/db/db_backup
sed -i 's/sawaya/[your system username]/g' dumpfile
```

##### Import the Database #####

This will set up the actual tables and sequences used by QC:

```
psql qfp < dumpfile
```

#### Building binutils-gdb ####

This describes a global installation.  You may use **--prefix=[path]**
on the  configure command line to install to a different root
(such as /home/[your_login]/software). Of course, you'll want to adjust
your $PATH accordingly (i.e. cat 'export PATH=$PATH:/home/fred/software

Here are the steps:
{} = optional; [] = fill in the blank
```
cd [build location]
wget http://ftp.gnu.org/gnu/gdb/gdb-7.11.tar.gz
tar xf gdb-7.11.tar.gz
cd binutils-gdb-gdb-7.11-release [or something like that]
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
* cd qfp/perpVects
* make -f MakefileQD -j 12
* runQD [test1] [test2] [precision1] [sort1] [precision2] [sort2]

