# FLiT Configuration File

The FLiT configuration file is called `flit-config.toml`.  It is written in the
[TOML](https://github.com/toml-lang/toml) specification.

Here is an example configuration file:

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
