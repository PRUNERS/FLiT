#Automatic Test Input Generation
The input generator is a script which can produce random input for a test, and given two sets of compile flags, find divergence points between two compiles. This is useful for creating default test input.

To build the tool simply run:

	1. cd QFP/inputGen
	2. make

This will make the `inputGen` executable as well as all the tests in the `QFP/qfpc/tests` directory, this creates the list of tests for the tool. In the `makefile` is listed the compiler and flags used, which can be changed and recompiled to test different optimizations. The executable is well documented.

```
>./inputGen --help

Usage:
  inputGen [-h]
  inputGen --list-tests
  inputGen [-f|-d|-e|-a] [-m MAX_TRIES] [-i NUM_DIVERGENT_INPUTS] [-o FILENAME]
           [-r RAND_TYPE] (--test TEST_NAME ... | --all-tests)

Description:
  Runs the particular tests under the optimization given in compilation.  The
  test input is randomly generated floating point numbers.  It runs until the
  number of desired divergences are discovered between the unoptimized code
  and the optimized code.

Required Arguments:
  One of the following must be used since we need at least one test to run.
  The --test option can be specified more than once to specify more than one
  test.

  --test TEST_NAME
      The test to run.  For a list of choices, see --list-tests.

  --all-tests
      Run all tests that are returned from --list-tests.

Optional Arguments:

  -f, --float
      Run each test only with float precision.  This is the default.

  -d, --double
      Run each test only with double precision.

  -e, --long-double
      Run each test only with extended precision, also called long double.

  -a, --all-types
      Run each test with all precisions.

  -i NUM_DIVERGENT_INPUTS, --divergent-input-count NUM_DIVERGEN_INPUTS
      How many divergent inputs to attempt to identify and report.  The
      default value is 10.

  -m MAX_TRIES, --max-tries MAX_TRIES
      How many tries before giving up on search for the specified numer of
      divergent inputs.  The default value is 1,000.

  -o FILENAME, --output FILENAME
      Where to output the report.  By default it outputs to stdout.

  -r RAND_TYPE, --rand-type RAND_TYPE
      The type of random number distribution to use.  Choices are:
      - fp: Uniform over the floating-point number space
      - real: Uniform over the real number line, then projected to
        floating-point space
      The default is "fp".

```