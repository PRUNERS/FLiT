# Available Compiler Flags

[Prev](flit-configuration-file.md)
|
[Table of Contents](README.md)
|
[Next](writing-test-cases.md)

Convenient TOML lists:

* [GCC](#gcc)
* [Clang](#clang)
* [Intel](#intel)


## All Flags

In your configuration file `flit-config.toml` (see [FLiT Configuration
File](flit-configuration-file.md)), you specify compiler flags for each of the
compilers.  Only one compiler flag will be used with one optimization level.
If you want to have specific flag combinations, you can place it in the list,
such as `"-mavx2 -mfma -funsafe-math-optimizations"`.  Below is the original
default list for the supported compilers:

| Flag                          |  GCC  | Clang | Intel |
| ----------------------------- |:-----:|:-----:|:-----:|
| `-fassociative-math`          |   X   |   X   |       |
| `-fcx-fortran-rules`          |   X   |       |       |
| `-fcx-limited-range`          |   X   |       |   X   |
| `-fexcess-precision=fast`     |   X   |   X   |       |
| `-fexcess-precision=standard` |       |   X   |       |
| `-ffinite-math-only`          |   X   |   X   |       |
| `-ffloat-store`               |   X   |   X   |   X   |
| `-ffp-contract=on`            |   X   |   X   |       |
| `-fma`                        |       |       |   X   |
| `-fmerge-all-constants`       |   X   |   X   |   X   |
| `-fno-trapping-math`          |   X   |   X   |       |
| `-fp-model fast=1`            |       |       |   X   |
| `-fp-model fast=2`            |       |       |   X   |
| `-fp-model=double`            |       |       |   X   |
| `-fp-model=extended`          |       |       |   X   |
| `-fp-model=precise`           |       |       |   X   |
| `-fp-model=source`            |       |       |   X   |
| `-fp-model=strict`            |       |       |   X   |
| `-fp-port`                    |       |       |   X   |
| `-freciprocal-math`           |   X   |   X   |       |
| `-frounding-math`             |   X   |   X   |   X   |
| `-fsignaling-nans`            |   X   |   X   |       |
| `-fsingle-precision-constant` |       |   X   |   X   |
| `-ftz`                        |       |       |   X   |
| `-funsafe-math-optimizations` |   X   |   X   |       |
| `-march=core-avx2`            |       |   X   |   X   |
| `-mavx`                       |   X   |   X   |   X   |
| `-mavx2 -mfma`                |   X   |   X   |   X   |
| `-mfpmath=sse -mtune=native`  |   X   |   X   |   X   |
| `-mp1`                        |       |       |   X   |
| `-no-fma`                     |       |       |   X   |
| `-no-ftz`                     |       |       |   X   |
| `-no-prec-div`                |       |       |   X   |
| `-prec-div`                   |       |       |   X   |

For your convenience, here are toml-style lists that can be copied into your
`flit-config.toml` file directly:

## GCC

**Note:** in some versions of python-toml, there is a parsing bug when a list
has an empty string in the middle.  So simply put it at the end without a
comma.  This has been fixed in the latest version of python-toml.

You will likely want the empty string in your search space as it represents the
absence of any switches.

```toml
switches_list = [
  '-fassociative-math',
  '-fcx-fortran-rules',
  '-fcx-limited-range',
  '-fexcess-precision=fast',
  '-ffinite-math-only',
  '-ffloat-store',
  '-ffp-contract=on',
  '-fmerge-all-constants',
  '-fno-trapping-math',
  '-freciprocal-math',
  '-frounding-math',
  '-fsignaling-nans',
  '-funsafe-math-optimizations',
  '-mavx',
  '-mavx2 -mfma',
  '-mfpmath=sse -mtune=native',
  ''
]
```

## Clang

```toml
switches_list = [
  '-fassociative-math',
  '-fexcess-precision=fast',
  '-fexcess-precision=standard',
  '-ffinite-math-only',
  '-ffloat-store',
  '-ffp-contract=on',
  '-fmerge-all-constants',
  '-fno-trapping-math',
  '-freciprocal-math',
  '-frounding-math',
  '-fsignaling-nans',
  '-fsingle-precision-constant',
  '-funsafe-math-optimizations',
  '-march=core-avx2',
  '-mavx',
  '-mavx2 -mfma',
  '-mfpmath=sse -mtune=native',
  ''
]
```

## Intel

```toml
switches_list = [
  '--use_fast_math',
  '-fcx-limited-range',
  '-ffloat-store',
  '-fma',
  '-fmerge-all-constants',
  '-fp-model fast=1',
  '-fp-model fast=2',
  '-fp-model=double',
  '-fp-model=except',
  '-fp-model=extended',
  '-fp-model=precise',
  '-fp-model=source',
  '-fp-model=strict',
  '-fp-port',
  '-frounding-math',
  '-fsingle-precision-constant',
  '-ftz',
  '-march=core-avx2',
  '-mavx',
  '-mavx2 -mfma',
  '-mfpmath=sse -mtune=native',
  '-mp1',
  '-no-fma',
  '-no-ftz',
  '-no-prec-div',
  '-prec-div',
  ''
]
```

[Prev](flit-configuration-file.md)
|
[Table of Contents](README.md)
|
[Next](writing-test-cases.md)
