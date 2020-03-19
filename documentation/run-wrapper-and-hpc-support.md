# Run Wrapper and HPC Support

[Prev](cuda-support.md)
|
[Table of Contents](README.md)
|
[Next](compiling-your-tests.md)


Within the `custom.mk` file, there is a variable called `RUN_WRAPPER`.  This
`RUN_WRAPPER` is a program that is to wrap around all test executable runs.
For example, if you set

```make
RUN_WRAPPER = @echo
```

then the tests will not be executed, but instead will only be printed to show
what would be executed.  Alternatively, you can create your own custom wrapper
script such as `wrapper.sh`:

```bash
#!/bin/bash

echo "$@" >> test-execution.log
time "$@"
```

It used to be this approach was used to integrate with Slurm.  Instead, we
prefer you perform an `sbatch` or `salloc` before-hand, and within that
allocation, you run the FLiT tests.  If you want parallelism in the FLiT tests
themselves, then call `flit_mpi_main()`.  If your program has different
parallelism than MPI, this function can still be used with `srun` instead of
`mpirun`.

You can use this wrapper to do other things like monitor memory usage or use an
automated checker for memory leaks.  Overall it can be a useful feature to tie
in your own custom functionality on top of the test executables.


[Prev](cuda-support.md)
|
[Table of Contents](README.md)
|
[Next](compiling-your-tests.md)
