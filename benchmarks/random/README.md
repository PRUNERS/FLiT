# Random Benchmark

This benchmark is to test that the C++ `<random>` header is portable accross
compilers.  My guess is that they will be since I suspect they are implemented
in the GLibC, but I'm not sure.  Hence the tests.  This can give us confidence
that this is indeed safe to depend on in FLiT tests.

We also test the `rand()` and `srand()` functions that are part of the standard
`<cstdlib>` header file.  However, we do not test the non-portable `random()`,
`srandom()`, `initstate()`, and `setstate()` functions that are part of the
POSIX standard, but are not part of the C++ standard.
