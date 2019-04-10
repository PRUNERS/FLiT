# MPI Support

[Prev](writing-test-cases.md)
|
[Table of Contents](README.md)
|
[Next](cuda-support.md)


FLiT has built-in support for tests that utilize MPI.  Before going into the
details of this, a word of caution:

_**Warning**: FLiT requires test results to be exactly deterministic in order to
function properly and to give meaningful and trustworthy results.  By using
MPI, it is highly likely that nondeterminism can be introduced by concurrency.
It is the responsability of the test engineer to ensure the code under test is
deterministic.  FLiT gives no guarantee for nondeterministic code._


## Compiling FLiT with MPI Support

To compile FLiT with MPI support is actually not necessary.  The actual
compiled FLiT shared library has no need to contain any MPI code, so there is
no need to recompile FLiT to have MPI support.  This also means that an MPI
implementation does not need to be installed in order to compile and install
FLiT.  This may change in future versions, but as of now, you can simply enable
MPI support in the tests.

This has an added benefit of not requiring to recompile FLiT if you change your
implementation of MPI you want to use.  FLiT will not need to be recompiled,
just the tests.


## Enabling MPI Support

To enable MPI support, you simply need to add (or edit) two lines in your
`flit-config.toml` file (see [FLiT Configuration
File](flit-configuration-file.md)).  For example:

```toml
[run]
enable_mpi = true
mpirun_args = '-n 4'
```

With the enabling of MPI, the generated Makefile will automatically pull
compiler and linker flags from `mpic++`.  The `mpic++` executable will be
pulled from the system `PATH` variable, so simply make sure your `PATH` is
pointing to the desired executable.

The `mpirun_args` is a place where you can specify any options you wish for
`mpirun`.  Again the `mpirun` executable is expected to be found using your
system `PATH` variable.


## Initializing and Finalizing the MPI Environment

Since the MPI standard recommends to initialize the MPI environment as close to
the beginning of the application as possible, when MPI support is enabled for
your tests, the FLiT framework will automatically call `MPI_Init()` and
`MPI_Finalize()` for you.  Please do not call these functions from within your
tests.


## MPI Information

There is a global variable you can have access to from within your tests that
FLiT provides, if you'd like.  You can access this functionality directly
through the MPI interface instead.  The global variable is `flit::mpi`, and it
is a pointer to a struct that can give you the world size, your world rank, and
a few other things.  See `src/MpiEnvironment.h` in the FLiT source code for
more details.  One of the benefits of this approach is that you can use this
global pointer even if MPI is not enabled and you can have logic based on the
rank and size of the MPI world.


## Test Return Values

Under FLiT, you can write tests using MPI with multiple processes (as long as
the result is deterministic).  To prevent confusion with multiple test return
values, one from each process, FLiT will ignore the test result value from all
processes except for the one with rank 0.  Therefore, all calculated results
must be communicated to rank 0 in order to have it be captured by the FLiT
framework.

If you find this does not fit with your use case, please submit an
[issue](https://github.com/PRUNERS/FLiT/issues/new?template=feature_request.md)
on GitHub.  For example, it may be that users may want to configure whether
each result is captured individually and checked at test time, process by
process.  However, That approach will not work in all cases, such as the
producer-consumer model where processes may end up getting different data and
tasks.  For now, enforcing users to communicate with rank 0 is the most
straightforward way to ensure determinism.


## Conditional Compilation

If you want your tests to conditionally use MPI, there is a macro-defined
variable you can condition off of to disable or enable certain code.  That
variable is `FLIT_USE_MPI`.  For example

Note: even when MPI is disabled, the global `flit::mpi` variable is still
available.

```c++
T grid[1024][1024];
int nrows = 1024 / mpi->size;
for (int row = mpi->rank * nrows; row < (mpi->rank + 1) * nrows; row++) {
  for (int col = 0; col < 1024; col++) {
    // some computation to populate or update grid
    // ...
  }
}
#ifdef FLIT_USE_MPI
MPI_Send(...);
#endif // FLIT_USE_MPI
```


[Prev](writing-test-cases.md)
|
[Table of Contents](README.md)
|
[Next](cuda-support.md)
