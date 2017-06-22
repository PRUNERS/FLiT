# FLiT Configuration File

[Prev](flit-command-line.md)
|
[Table of Contents](README.md)
|
[Next](available-compiler-flags.md)

The FLiT configuration file is called `flit-config.toml`.  It is written in the
[TOML](https://github.com/toml-lang/toml) specification.

Here is a full example configuration file:

```toml
[database]

username = 'mbentley'
address = 'localhost'
type = 'postgres'
port = 5432

[[hosts]]

name = 'my.hostname.com'
flit_path = '/usr/bin/flit'
config_dir = 'project/flit-tests'

[hosts.ground_truth]

compiler = 'g++'
optimization_level = '-O0'
switch = ''

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

[hosts.ground_truth]

compiler = 'Intel'
optimization_level = '-O0'
switch = ''

  [[hosts.compilers]]

  binary = 'icpc'
  name = 'intel'
  type = 'intel'
  optimization_levels = [
    '-O0',
  ]
  switches = [
    '',
  ]
```

We will now go through the elements of this configuration file piece by piece
and explain what they represent.

_Note: there are no default values for these fields.  You must specify all
fields for an entry to be valid.  If in doubt, you can use [flit
check](flit-command-line.md#flit-check) to verify your configuration file is
valid._

```toml
[database]

username = 'mbentley'
address = 'localhost'
type = 'postgres'
port = 5432
```

Above we specify the information for the database connection.  Since
it is not very secure to store passwords in the configuration file,
you will be prompted for the password at the time of execution.  Right
now, `postgres` is the only database type that is supported, but more
are to come.

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

For any variable that specifies a relative path, the relative path is always
with respect to the current user's home directory.

```toml
[hosts.ground_truth]

compiler = 'g++'
optimization_level = '-O0'
switch = ''
```

Each host has a ground truth run to use as comparisons when doing analysis.
The `compiler` field here needs to match the `name` of the compiler specified
later in the `[[hosts.compilers]]`.  Also, the `optimization_level` and
`switch` need to be available in the `optimization_levels` and `switches` for
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

Here we specify the first compiler for the first host.  Since binary here is a simple name, it will get the executable `g++` from the system `PATH`.  If you really mean you want to have a compiler that is located in your home directory, you can do `./g++`.

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

[hosts.ground_truth]

compiler = 'Intel'
optimization_level = '-O0'
switch = ''

  [[hosts.compilers]]

  binary = 'icpc'
  name = 'intel'
  type = 'intel'
  optimization_levels = [
    '-O0',
  ]
  switches = [
    '',
  ]
```

Here it is demonstrated that you can specify another host.  This one is called
`other.hostname.com` with a single compiler named `intel`.

[Prev](flit-command-line.md)
|
[Table of Contents](README.md)
|
[Next](available-compiler-flags.md)
