# MPI Support

[Prev](flit-helpers.md)
|
[Table of Contents](README.md)
|
[Next](cuda-support.md)


FLiT has built-in support for tests that utilize MPI.  Before going into the
details of this, a word of caution:

_**Warning**: FLiT requires test results to be exactly deterministic in order to
function properly and to give meaningful and trustworthy results.  By using
MPI, it is highly likely that nondeterminism can be introduced by concurrency.
It is the responsibility of the test engineer to ensure the code under test is
deterministic.  It may work, but FLiT gives no guarantee for nondeterministic
code._


## Enabling MPI Support

To enable MPI support, you simply need to add (or edit) two lines in your
`flit-config.toml` file (see [FLiT Configuration
File](flit-configuration-file.md)).  For example:

```toml
[run]
enable_mpi = true
```

With the enabling of MPI, the generated Makefile will automatically pull
compiler and linker flags from `mpic++`.  The `mpic++` executable will be
pulled from the system `PATH` variable, so simply make sure your `PATH` is
pointing to the desired executable.

The arguments to `mpirun` are specified in your test itself.  To run MPI tests,
you will need to use the `flit::call_mpi_main()` function after registering a
main-like function with `FLIT_REGISTER_MAIN()`.  By "main-like" function, we
mean a function taking in an `int` and a `char**` and returning and `int`.
Please see [Writing Test Cases: Writing MPI
Tests](writing-test-cases.md#writing-mpi-tests) for more information.

Alternatively, you can specify your `mpic++` executable file as your compiler
in the `[[compiler]]` section of `flit-config.toml`.  If you do this, you can
safely set `enable_mpi = false`, since that feature is only used for
compilation.


## Writing MPI Tests

The only way to write MPI tests is to have your test call a main-like function
that handles all of the MPI requirements.  See [Writing Test
Cases](writing-test-cases.md#writing_mpi_tests) for more information.

This is the only supported way of doing MPI from FLiT.  The MPI standard
requires the user to call `MPI_Init()` as close to the beginning of the
application as possible and to call `MPI_Finalize()` as close to the end of the
application and after any MPI calls.  Calling these methods from within the
FLiT test directly combined with calling the flit test executable using
`mpirun` will likely have undesired behavior.


[Prev](flit-helpers.md)
|
[Table of Contents](README.md)
|
[Next](cuda-support.md)
