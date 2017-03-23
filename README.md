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

Additionally, we've provided the capability to __time__ the test suite, generating
a time plot per test, comparing both the result divergence and the timing for
each compiler flag.

It consists of the following components:

* a test infrastructure in the form of c++ code, where additional tests
  are easily added
* a dynamic make system to generate diverse executables
* an execution disbursement system
* a SQL database for collecting results
  * a collection of queries to help the user understand results
  * some data analysis tools, providing visualization of results

Contents:

* [Prerequisites and setup](#prerequisites-and-setup)
  * [Tested host configurations](#tested-configurations)
  * [Setup launch host](#setup-launch-host)
  * [Configuring DB host](#configuring-db-host)
  * [Setting up ssh keys](#setting-up-ssh-keys)
  * [Specifying host list](#specifying-host-list)
  * [Configuring tests](#configuring-tests)
	* [Adding tests](#adding-tests)
  * [Choosing compilers and flags](#choosing-compilers-and-flags)
* [Running FLiT](#running-flit)
* [Viewing the Results](#view-test-results)

## Prerequisites and Setup ##

FLiT is designed to build and excute its test suite on a variety of hosts and
compilers.  There are many ways you can configure FLiT to run on different hosts.
There are 3 different host roles that are used for running a FLiT test, and 
it is not required that they be different machines (i.e. you can do a run on
a single host, allowing it to play all 3 roles). 

They are:

* Launch host - this is the machine that you launch the tests from, and 
  clone the git repository to
* Worker hosts - one or more hosts to run the test suite on
* Database host - this is where the data is imported to the database, and
  where reports are generated (divergence and timing plots)
* Additionally, when using Slurm on a cluster, there may be a 'staging' host, where the slurm script is launched from
  
Without any configuration, FLiT will __run out of the box__. The only requirements
are:

* the __git__ tool [on the launch host]
* a basic __Unix__ family operating system, providing common tools such as 
  __bash__, __tar__, __python3__, __make__ [for all hosts]
* Presence of g++, clang++ or icpc (Intel c++ compiler) [on the work node(s)]
* A modern package management system [needed to configure the database host]
  * apt
  * apt-get
  * pacman
  * yum
  * brew
  
Additionally, FLiT provides support for many HPC cluster configurations, where
jobs may be submitted with a script.  We provide sample scripts for using
the __Slurm__ resource manager.

### Tested Configurations ###

Launch, DB host:

* Ubuntu 16.x
* Python 3.5.2
* git 2.7.4

Worker hosts:

* Ubuntu 16.x
* CentOS 6.x+
* Python 3.x

### Setup launch host ###

To setup the launch host, it is only necessary to clone this project into 
a file space you may run from.

`
git clone [the git url]
`

### Configuring DB host ###

FLiT provides automation to take care of this task.  You need to 
copy a single script from the launch host to the db host and then
run it (replacing _[username]_ and _[db\_hostname]_ as appropriate)

```
scp FLiT/db/InstallFlitDB.sh [username]@[db_hostname]:
ssh [username]@[db_hostname] 
./InstallFlitDB.sh
exit
```

### Setting up ssh keys ###

If you use a cluster with a resource manager script (such as ones we 
provide, like _kingspeak\_cpu\_startup_), we have no way to copy 
results out of whatever host the cluster scheduler chooses to run
your job, so it will be necessary to configure SSH keys, allowing
the selected worker to copy files to the _db\_host_.  These are 
the steps needed (replacing _[cluster login node]_ and 
_[cluster user name]_ as appropriate):

Log into the cluster, and go to your _.ssh_ directory:
```
ssh [cluster user name]@[cluster login node]
[enter password]
ls .ssh
```
If you already have the files _id\_rsa_ and _id\_rsa.pub_, skip 
to the next step. Otherwise, generate a new ssh key-pair:

```
ssh-keygen -b 4096
[enter]
[enter]
[enter]
exit
```

Finally, copy the public key over to your chosen _db\_host_, and add it to the
_authorized\_keys_ file:

```
scp [cluster_username]@[cluster_hostname]:.ssh/id_rsa.pub [db_username]@[db_hostname]:
[enter passwords]
ssh [db_username]@[db_hostname]
cat id_rsa.pub >> .ssh/authorized_keys
exit
```

### Specifying host list ###

It is necessary, for each worker host and the db host, to add
details to the file _FLiT/scripts/hostfile.py_.  By default, this file
is configured to run all roles on the same host that you cloned this
project to.

IF you want to use different machines for the database server or workers,
follow the instructions in the file to add them.  You may add as many
worker hosts as you like (you must have an account and ssh access to them).

### Configuring tests ###

The test files exist in _FLiT_/src/tests.  You may just run the default
tests, in which case you don't have to do anything. 

To prevent tests from running, just remove them from the _tests_ directory.

#### Adding tests ####

All litmus tests that are included in general execution are located in the _FLiT/src/tests_ directory. For a test to be included in a run it only need be present in this directory. Tests must be in a templated C++ class which extends `QFPTest::TestBase<T>`.

##getInputsPerRun
This function returns the number of arguments the code under tests uses.

##getDefaultInput
This function returns a `QFPTest::TestInput`, which is a vector like object whose `.vals` field holds a list of inputs. Note that more than one set of inputs may be placed in this field.

##run_impl
This function is the actual code under test. The argument is a `QFPTest::TestInput` with length of the `.vals` field equal to the number of inputs requested per run. The return is a vector of two values representing the score of the test. This does not need to represent 'good' or 'bad' results, as the score is used to classify different compilations.

##REGISTER_TEST
Just as it sounds, this function is used to register the test in the framework. It's argument is the class created for the test.

#CUDA Tests
The main setup for CUDA tests is the same, with the exception that the `run_impl` function is a CUDA kernel which is placed in the framework though `getKernel()` and it has a second argument which is treated as an inout variable with fields `.s1` and `.s2` to hold the scores.

#Integrating into the test framework
The test cpp file must be placed in `QFP/qfpc/tests` for it to be included in subsequent runs. No other changes are required.

#Starting code
This is a simple source file to start your own tests.

```
#include "test_base.hpp" 
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

// These functions are not needed, but it is nice to break out the core computation being done
template <typename T>
DEVICE
T 
my_cuda_test_core(const T input_1) // Number of arguments can be changed
{
    return input_1; // Cuda test code
}

template <typename T>
T 
my_cpp_test_core(const T input_1) // Number of arguments can be changed
{
    return input_1; // Cpp test code
}


template <typename T>
GLOBAL
void 
my_cuda_kern(const QFPTest::CuTestInput<T>* tiList, QFPTest::CudaResultElement* results){
    using namespace CUHelpers;
#ifdef __CUDA__
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
    auto idx = 0;
#endif
    T input_1 = tiList[idx].vals[0]; // Extract arguments from vals

    // Loops or other structures can be used here such as in TrianglePSylv.cpp  
    double score = my_cuda_test_core(input);

    results[idx].s1 = score;
    results[idx].s2 = 0.0;
}

template <typename T>
class MyTest: public QFPTest::TestBase<T> {
public:
    MyTest(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}

    // This must be changed to match the number of arguments used in the test code
    virtual size_t getInputsPerRun() { return 1; } 

    virtual QFPTest::TestInput<T> getDefaultInput() {
        QFPTest::TestInput<T> ti;
        ti.vals = { 6.0 };
        return ti;
    }

protected:
    virtual
    QFPTest::KernelFunction<T>* getKernel() {return my_cuda_kern; }
    virtual
    QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
        T input_1 = ti.vals[0]; // Extract arguments from vals
        
        // Loops or other structures can be used here such as in TrianglePSylv.cpp  
        double score = my_cpp_kern(input_1);

        return {score, 0.0};
    }

protected:
    using QFPTest::TestBase<T>::id;
};

REGISTER_TYPE(MyTest)
```

### Choosing compilers and flags ###

The Makefile system will use any _clang++_, _g++_ and _icpc_ compiler
that it finds on your __$PATH__.  It will also try to run __CUDA__ tests
if _nvcc_ is found on your __$PATH__.

You may add more flags by modifying the _FLiT/Makefile.switches_ file.

To do this, you must make two changes:

* add a name and literal flag pair:
``` SOMEFLAG -abc```

and then add it to the list for the compiler(s) that it belongs to, 
one of: 

* SWITCHES_GCC
* SWITCHES_CLANG
* SWITCHES_INTEL

## Running FLiT ##

To run flit, you must run the script _FLiT/scripts/run_all.py_, and give it
a string that you'd like to describe the run:

```
cd FLiT/scripts
./run_all.py "this is my first FLiT run"
```

It may take a while, but after the script returns control to your console
you should have some results to view.

## View test results ##

After a successful run, there will be __.pdf__ files on the database host
that allow you to visualize the divergencies in the tests.  They will
be located in _flit_data_/reports.



