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

_Note: there are no default values for these fields.  You must specify all
fields for an entry to be valid.  If in doubt, you can use [flit
check](flit-command-line.md#flit-check) to verify your configuration file is
valid._

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

compiler_name = 'clang++'
optimization_level = '-O2'
switches = '-mavx'
```

Each host has a developer build that can be performed.  The purpose of the
developer build is to try out the tests that you have developed without
committing yourself to a full FLiT run.

The `compiler` field here needs to match the `name` of one of the compilers specified
later in the `[[hosts.compilers]]` list.  The `optimization_level` and
`switches` need not be in the `optimization_levels` and `switches` for
the compiler with the matching name.

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
  type = 'gcc'
  optimization_levels = [
    '-O0',
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx2 -mfma',
  ]
```

Here we specify the first compiler for the first host.  Since binary here is a
simple name, it will get the executable `g++` from the system `PATH`.  If you
really mean you want to have a compiler that is located in your home directory,
you can do `./g++`.

The `type` parameter can be one of

* `gcc`
* `clang`
* `intel`
* `cuda`

The `optimization_levels` and `switches` will be combined as a cartesian
product and each possible pairing will become a compilation performed by FLiT.
For a list of all possible flags for each compiler type, see [Available
Compiler Flags](available-compiler-flags.md).

```toml
  [[hosts.compilers]]

  binary = 'my-installs/g++-7.0/bin/g++'
  name = 'g++-7.0'
  type = 'gcc'
  optimization_levels = [
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx2 -mfma',
    '-funsafe-math-optimizations',
  ]
```

Here it is demonstrated that you can specify a second version of `g++`
with different optimization levels and switches from the first
version.  It is simply required that the compiler name be unique for
this host.

```toml
  [[hosts.compilers]]

  binary = 'clang++'
  name = 'clang'
  type = 'clang'
  optimization_levels = [
    '-O0',
    '-O1',
    '-O2',
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx',
    '-fexcess-precision=fast',
    '-ffinite-math-only',
    '-mavx2 -mfma',
    '-march=core-avx2',
  ]
```

We also specify a third compiler `clang++`, again with different flags.  So for
the host `my.hostname.com`, we have three compilers configured: `g++`,
`g++-7.0`, and `clang`.

```toml
[[hosts]]

name = 'other.hostname.com'
flit_path = 'my-installs/flit/bin/flit'
config_dir = 'project/flit-tests'

[hosts.dev_build]

compiler_name = 'intel-17.0'
optimization_level = '-O2'
switches = '-mavx'

[hosts.ground_truth]

compiler_name = 'intel-17.0'
optimization_level = '-O0'
switch = ''

  [[hosts.compilers]]

  binary = 'icpc'
  name = 'intel-17.0'
  type = 'intel'
  optimization_levels = [
    '-O0',
  ]
  switches = [
    '',
  ]
```

Here it is demonstrated that you can specify another host.  This one is called
`other.hostname.com` with a single compiler named `intel-17.0`.

## Full Configuration File

Combining all of the above sections together, here is the full example configuration file:

```toml
[database]

type = 'sqlite'
filepath = 'results.sqlite'

[[hosts]]

name = 'my.hostname.com'
flit_path = '/usr/bin/flit'
config_dir = 'project/flit-tests'

[hosts.dev_build]

compiler_name = 'clang++'
optimization_level = '-O2'
switches = '-mavx'

[hosts.ground_truth]

compiler_name = 'g++'
optimization_level = '-O0'
switches = ''

  [[hosts.compilers]]

  binary = 'g++'
  name = 'g++'
  type = 'gcc'
  optimization_levels = [
    '-O0',
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx2 -mfma',
  ]

  [[hosts.compilers]]

  binary = 'my-installs/g++-7.0/bin/g++'
  name = 'g++-7.0'
  type = 'gcc'
  optimization_levels = [
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx2 -mfma',
    '-funsafe-math-optimizations',
  ]

  [[hosts.compilers]]

  binary = 'clang++'
  name = 'clang'
  type = 'clang'
  optimization_levels = [
    '-O0',
    '-O1',
    '-O2',
    '-O3',
  ]
  switches = [
    '',
    '-fassociative-math',
    '-mavx',
    '-fexcess-precision=fast',
    '-ffinite-math-only',
    '-mavx2 -mfma',
    '-march=core-avx2',
  ]

[[hosts]]

name = 'other.hostname.com'
flit_path = 'my-installs/flit/bin/flit'
config_dir = 'project/flit-tests'

[hosts.dev_build]

compiler_name = 'intel-17.0'
optimization_level = '-O2'
switches = '-mavx'

[hosts.ground_truth]

compiler_name = 'intel-17.0'
optimization_level = '-O0'
switch = ''

  [[hosts.compilers]]

  binary = 'icpc'
  name = 'intel-17.0'
  type = 'intel'
  optimization_levels = [
    '-O0',
  ]
  switches = [
    '',
  ]
```


[Prev](flit-command-line.md)
|
[Table of Contents](README.md)
|
[Next](available-compiler-flags.md)
