This is a test suite that was translated from the Polybench benchmark suite.
The original can be found [here](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/).

Each source file has a main class extending the FLiT test class while also
parameterizing the size of input arrays for the given polyhedral kernel.
The number of parameters varies from benchmark to benchmark.

For convenience these classes can be specialized and entered into the FLiT tests
by using the REGISTER_*N* macros where *N* is the number of parameterized arguments
not including the float type used by FLiT. This macro takes the name of the base
class as the first argument and numbers for all other arguments.

To run the suite as is, with flit in your path:
```
flit update
flit make
```

Manifest:
	-custom.mk: A FLiT generated makefile used to add additional flags to
	            the compilations.
	-flit-config.toml: A FLiT generated configuration for integration into
	                   FLiT with a default setup for ibriggs on the machine
			   fractus.
	-README.md: This file.
	-tests/*.cpp: Kernels from the Polybench suite.
	-tests/polybench_utils.hpp: Convenience functions and macros.