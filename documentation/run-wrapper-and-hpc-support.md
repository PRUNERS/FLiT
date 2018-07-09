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

This will give you freedom to log, to time, or to even then run them in
parallel with Slurm (using `srun` or `sbatch`).  You can also do things like
monitor memory usage or use an automated checker for memory leaks.  Overall it
can be a useful feature to tie in your own custom functionality on top of the
test executables.

Note, it is not good to launch too many small jobs using srun, but the amount
of jobs that FLiT generates is only approximately 300 or so, depending on how
many compilers are found in your PATH.  That may or may not be too many for
your needs.


[Prev](cuda-support.md)
|
[Table of Contents](README.md)
|
[Next](compiling-your-tests.md)
