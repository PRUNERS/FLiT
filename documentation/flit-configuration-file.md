# FLiT Configuration File

[Prev](flit-command-line.md)
|
[Table of Contents](README.md)
|
[Next](available-compiler-flags.md)

The FLiT configuration file is called `flit-config.toml`.  It is written in the
[TOML](https://github.com/toml-lang/toml) specification.

Here I have an example configuration file.  I will go through it section by
section.  If you want to see the full configuration file, it is at the end of
this page.

_Note: this is not just an example configuration file, it also contains all of
the default values too.  When your project gets initialized using_
[`flit init`](flit-command-line.md#flit-init)_,
that configuration file will also contain the default values.  You are welcome
to delete or comment out the values, which will cause the default values to be
used._

```toml
[database]

type = 'sqlite3'
filepath = 'results.sqlite'
```

Above we specify the information for the database.  Postgres used to be
supported, but has been replaced with SQLite3.  For now, only `sqlite3` is
supported for the `type`.  The only thing that needs to be specified is
`filepath`, which can be a relative or absolute path.

```toml
[run]

timing = true
timing_loops = -1
timing_repeats = 3
enable_mpi = false
mpirun_args = ''
```

Here we have information about how to execute the tests.  More specifically
some options that control the timing.  You can turn timing off entirely, or you
can change the defaults of the timing if you find it is taking too long to run
with the default timing procedure.

These options are only for the full run done by either calling `flit make` or
`make run`.  Some of these options may be used by `flit bisect` as well, but
not the timing ones since `flit bisect` does not care so much about timing.

* `timing`: `false` means to turn off the timing feature.  The full test run
  will be much faster with this option.  This is related to the `--timing` and
  `--no-timing` flags from the [test executable](test-executable.md#Timing).
* `timing_loops`: The number of loops to run before averaging.  For values less
  than zero, the amount of loops to run will be automatically determined.  This
  is the same as the `--timing-loops` option flag from the [test
  executable](test-executable.md#Timing).
* `timing_repeats`: How many times to repeat timing.  The minimum timing value
  will be kept.  This is the same as the `--timing-repeats` option flag from
  [test executable](test-executable.md#Timing).
* `enable_mpi`: Turns on compiling and running tests with MPI support.  See the
  [MPI Support](mpi-support.md) page for more information.
* `mpirun_args`: Arguments to pass to `mpirun`.  This is where you specify how
  many processes to run, for example `-n 16` to run 16 instances of the tests
  under MPI.

**A note about MPI:** FLiT requires the tests to be deterministic.  If the
tests employ MPI, it is the test creator's responsibility to ensure that the
test produces identical results every time.

```toml
[[hosts]]

name = 'my.hostname.com'
flit_path = '/usr/bin/flit'
config_dir = 'project/flit-tests'
```

Now we start specifying information of the first host called `my.hostname.com`.
The `flit_path` is the path of the `flit` command-line tool at this host.  The
`config_dir` is the directory where this configuration file is located on the
host.  For now, it is expected that each machine already has a copy of the
tests with the same configuration file and that their location is documented
here.  If needed, this may be revisited.

For any variable under `hosts` that specifies a relative path, the relative
path is always with respect to the user's home directory.

```toml
[hosts.dev_build]

compiler_name = 'g++'
optimization_level = '-O2'
switches = '-funsafe-math-optimizations'
```

Each host has a developer build that can be performed.  The purpose of the
developer build is to try out the tests that you have developed without
committing yourself to a full FLiT run.

The `compiler` field here needs to match the `name` of one of the compilers
specified later in the `[[hosts.compilers]]` list.  The `optimization_level`
and `switches` need not be in the `optimization_levels` and `switches` for the
compiler with the matching name.

This does exactly one compilation with the compiler and flags that are
specified.  This compilation is done using `make dev` and generates an
executable called `devrun`.  You can call

```bash
./devrun --help
```

to see the options available for the test executable, such as only running a
particular test or outputting debugging information.

```toml
[hosts.ground_truth]

compiler_name = 'g++'
optimization_level = '-O0'
switches = ''
```

Each host has a ground truth run to use as comparisons when doing analysis.
The `compiler` field here needs to match the `name` of one of the compilers specified
later in the `[[hosts.compilers]]` list.  The `optimization_level` and
`switches` need not be in the `optimization_levels` and `switches` for
the compiler with the matching name.

```toml
  [[hosts.compilers]]

  binary = 'g++'
  name = 'g++'
```

Here we specify the first compiler for the first host.  Since binary here is a
simple name, it will get the executable `g++` from the system `PATH`.  If you
really mean you want to have a compiler that is located in your home directory,
you can do `./g++`.  The name field is a human-readable unique name for this
compiler.  For example, the binary field can have an absolute path where the
name can be something like `gcc-7.4`.

## Full Configuration File

Combining all of the above sections together, here is the full example (and
default) configuration file:

```toml
[database]

type = 'sqlite'
filepath = 'results.sqlite'


[run]

timing = true
timing_loops = -1
timing_repeats = 3

enable_mpi = false
mpirun_args = ''


[[hosts]]

name = 'my.hostname.com'
flit_path = '/usr/bin/flit'
config_dir = 'project/flit-tests'


[hosts.dev_build]

compiler_name = 'g++'
optimization_level = '-O2'
switches = '-funsafe-math-optimizations'


[hosts.ground_truth]

compiler_name = 'g++'
optimization_level = '-O0'
switches = ''


  [[hosts.compilers]]

  binary = 'g++'
  name = 'g++'
```


[Prev](flit-command-line.md)
|
[Table of Contents](README.md)
|
[Next](available-compiler-flags.md)
