# Random Benchmark

This benchmark is to test that the C++ `<random>` header is portable accross compilers.  My guess is that they will be since I suspect they are implemented in the GLibC, but I'm not sure.  Hence the tests.  This can give us confidence that this is indeed safe to depend on in FLiT tests.
