# Release Notes

[Table of Contents](README.md)
|
[Next](installation.md)

Release note versions:

- [v2.0-beta.2](#v2.0-beta.2)
- [v2.0-beta.1](#v2.0-beta.1)
- [v2.0-alpha.3](#v2.0-alpha.3)
- [v2.0-alpha.2](#v2.0-alpha.2)
- [v2.0-alpha.1](#v2.0-alpha.1)
- [v1.0.0](#v1.0.0)

## v2.0-beta.2

Released on 10 April 2019

**Highlights of Major Changes**

- TravisCI continuous integration fixed and working
- `flit bisect`
    - outputs results incrementally.
    - optimizations, such as memoizing and search space reduction
    - order findings by the magnitude of variability
    - add flags `--compile-only` and `--precompile-fpic`
    - additional bisect assertion for finding disjoint sets
    - add dependency of `pyelftools`
- Users specify the FLiT search space in `flit-config.toml` instead of being hard-coded
- `flit import`: add `--dbfile` flag to create an import database without a `flit-config.toml` file
- Remove `hosts` section from `flit-config.toml`
- Many bug fixes

**All Issues Addressed:**

- [#174](https://github.com/PRUNERS/FLiT/issues/174) (PR [#179](https://github.com/PRUNERS/FLiT/pull/179)): Have bisect output results incrementally
- [#175](https://github.com/PRUNERS/FLiT/issues/175) (PR [#181](https://github.com/PRUNERS/FLiT/pull/181)): Have function passed to `bisect_search` only take one argument
- [#182](https://github.com/PRUNERS/FLiT/issues/182) (PR [#185](https://github.com/PRUNERS/FLiT/pull/185), [#186](https://github.com/PRUNERS/FLiT/pull/186), [#187](https://github.com/PRUNERS/FLiT/pull/187)): Fix TravisCI continuous integration
- [#180](https://github.com/PRUNERS/FLiT/issues/180) (PR [#190](https://github.com/PRUNERS/FLiT/pull/190)): Memoize the bisect test function
- [#143](https://github.com/PRUNERS/FLiT/issues/143) (PR [#195](https://github.com/PRUNERS/FLiT/pull/195)): Fix NaN return values from litmus tests
- [#172](https://github.com/PRUNERS/FLiT/issues/172) (PR [#197](https://github.com/PRUNERS/FLiT/pull/197)): Order bisect findings by magnitude of variability
- [#200](https://github.com/PRUNERS/FLiT/issues/200) (PR [#201](https://github.com/PRUNERS/FLiT/pull/201)): Fix CSV file parsing
- [#120](https://github.com/PRUNERS/FLiT/issues/120) (PR [#206](https://github.com/PRUNERS/FLiT/pull/206)): Allow user-provided compilers in `flit-config.toml`
- (PR [#209](https://github.com/PRUNERS/FLiT/pull/209)): Add `--compile-only` and `--precompile-fpic` to `flit bisect`
- [#211](https://github.com/PRUNERS/FLiT/issues/211) (PR [#215](https://github.com/PRUNERS/FLiT/pull/215), [#223](https://github.com/PRUNERS/FLiT/pull/223)): Add `--dbfile` to `flit import` to import without a `flit-config.toml` file
- [#220](https://github.com/PRUNERS/FLiT/issues/220) (PR [#221](https://github.com/PRUNERS/FLiT/pull/221)): Have the test harness delete temporary directories even when exceptions are thrown
- [#217](https://github.com/PRUNERS/FLiT/issues/217) (PR [#219](https://github.com/PRUNERS/FLiT/pull/219)): Create a template for issues and pull requests (a GitHub-specific update)
- [#212](https://github.com/PRUNERS/FLiT/issues/212) (PR [#213](https://github.com/PRUNERS/FLiT/pull/213)): For bisect, if the Makefile errors out, print out the Makefile output for debugging
- [#225](https://github.com/PRUNERS/FLiT/issues/225) (PR [#226](https://github.com/PRUNERS/FLiT/pull/226)): For bisect, add assertion for finding disjoint sets (cannot continue if they overlap)
- [#230](https://github.com/PRUNERS/FLiT/issues/230) (PR [#232](https://github.com/PRUNERS/FLiT/pull/232)): Fix compilation if header files go missing
- [#229](https://github.com/PRUNERS/FLiT/issues/229) (PR [#231](https://github.com/PRUNERS/FLiT/pull/231)): Fix bisect deadlock where multiple updates to `ground-truth.csv` were attempted
- [#194](https://github.com/PRUNERS/FLiT/issues/194) (PR [#236](https://github.com/PRUNERS/FLiT/pull/236)): Fix the slow bisect search due to an explosion of templated function definitions
- [#238](https://github.com/PRUNERS/FLiT/issues/238) (PR [#241](https://github.com/PRUNERS/FLiT/pull/241)): Fix bisect search space problems.  Specifically, only search over strong and publicly exported function symbols.  This pull request added the dependency of `pyelftools`.
- [#233](https://github.com/PRUNERS/FLiT/issues/233) (PR [#243](https://github.com/PRUNERS/FLiT/pull/243)): Update the required version of `binutils` in the documentation
- [#239](https://github.com/PRUNERS/FLiT/issues/239) (PR [#242](https://github.com/PRUNERS/FLiT/pull/242)): Fix support for older versions of the Clang compiler (version 4)
- [#244](https://github.com/PRUNERS/FLiT/issues/244) (PR [#246](https://github.com/PRUNERS/FLiT/pull/246)); Remove the `hosts` section from `flit-config.toml`.  It was not used.
- [#249](https://github.com/PRUNERS/FLiT/issues/249) (PR [#250](https://github.com/PRUNERS/FLiT/pull/250)): Update documentation about how FLiT handles MPI tests.  Specifically how the test return values are handled.
- [#119](https://github.com/PRUNERS/FLiT/issues/119) (PR [#351](https://github.com/PRUNERS/FLiT/pull/351)): Allow the user to specify the list of optimization levels and flags in `flit-config.toml` for the FLiT search space.  No longer hard-coded.
- [#256](https://github.com/PRUNERS/FLiT/issues/256) (PR [#258](https://github.com/PRUNERS/FLiT/pull/258)): Fix bisect crash when an object file has an empty dwarf info
- [#257](https://github.com/PRUNERS/FLiT/issues/257) (PR [#259](https://github.com/PRUNERS/FLiT/pull/259)): Fix bisect crash when an object file has no dwarf info section


## v2.0-beta.1

Since FLiT has been stable in the alpha phase for some time and some other things have stabilized more (such as documentation and interface), we are happy to announce the move to beta status.

In this release, we have included the following pull requests:

[#115](https://github.com/PRUNERS/FLiT/pull/115), [#116](https://github.com/PRUNERS/FLiT/pull/116), [#128](https://github.com/PRUNERS/FLiT/pull/128), [#130](https://github.com/PRUNERS/FLiT/pull/130), [#131](https://github.com/PRUNERS/FLiT/pull/131), [#132](https://github.com/PRUNERS/FLiT/pull/132), [#135](https://github.com/PRUNERS/FLiT/pull/135), [#139](https://github.com/PRUNERS/FLiT/pull/139), [#142](https://github.com/PRUNERS/FLiT/pull/142), [#145](https://github.com/PRUNERS/FLiT/pull/145), [#147](https://github.com/PRUNERS/FLiT/pull/147), [#150](https://github.com/PRUNERS/FLiT/pull/150), [#154](https://github.com/PRUNERS/FLiT/pull/154), [#155](https://github.com/PRUNERS/FLiT/pull/155), [#156](https://github.com/PRUNERS/FLiT/pull/156), [#157](https://github.com/PRUNERS/FLiT/pull/157), [#158](https://github.com/PRUNERS/FLiT/pull/158), [#159](https://github.com/PRUNERS/FLiT/pull/159), [#160](https://github.com/PRUNERS/FLiT/pull/160), [#161](https://github.com/PRUNERS/FLiT/pull/161), [#167](https://github.com/PRUNERS/FLiT/pull/167), [#170](https://github.com/PRUNERS/FLiT/pull/170), [#176](https://github.com/PRUNERS/FLiT/pull/176)

This includes the following features:

- Bisect
    - Fully implemented and supported - assigns variability blame to individual source files and functions
    - Works with Intel linker problems
    - Autodelete intermediate generated files to reduce disk usage from going out of control
    - Autorun all differences found in a given database file
    - Unfortunately, this functionality is not adequately documented (see issue [#136](https://github.com/PRUNERS/FLiT/pull/136))
- Add MPI support within `flit-config.toml`
- Add timing parameters to `flit-config.toml`
- Add `--info` flag to test executable to show how it was compiled
- Rename results and database columns to be more understandable
- Add `plot_timing_histogram.py` for auto-generating histogram plots and update `plot_timing.py`
- Add benchmarks: polybench and random - benchmark examples of FLiT usage
- Add `uninstall` target to the top-level `Makefile`
- Remove CUDA completely, it was broken and cumbering the space
- Remove old unused files. The repository is more lean with all files relevant
- License notices on files
- Fix support for GCC 4.8 (had a compiler error in FLiT)


## v2.0-alpha.3

There was reported a bug in the installation making `flit import` not work. This has been fixed and a new alpha release is ready.


## v2.0-alpha.2

Many fixes in this update:

- Remove `TestInput` and `CUTestInput` types, replace with arrays and `std::vector`
- Disable unfinished flit command-line commands
- Remove sections of the `flit-config.toml` settings file that are unused
- Fix incremental build
- Add `--version` argument to `flit`
- Recursive generated `Makefile` allows for more scalability with project size and number of tests
- Add some tests for FLiT's infrastructure - not comprehensive, but some automated tests
- Fix timing functionality
- Add a time plot


## v2.0-alpha.1

Not everything for this new release is finished. The tool has been rearchitected to be easier to use within existing code bases. Many things have changed. Here are a few of them:

- A command-line tool called `flit` with subcommands
- Storage is into an SQLite database rather than Postgres (for simplicity and ease of installation)
- A configuration file, `flit-config.toml`, where you can specify your ground-truth compilation and your developer compilation.  Some of the values in this configuration file are not yet implemented to be used.
- An included `custom.mk` makefile to help you specify how to build your source files with the tests
- Turn FLiT into a shared library to be linked


## v.1.0.0

This release provides litmus tests, a reproducibility test framework, and analysis tools for the results of the tests.


[Table of Contents](README.md)
|
[Next](installation.md)
