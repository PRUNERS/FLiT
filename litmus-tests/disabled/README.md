# Disabled Litmus Tests

There are some litmus tests that are capable of being run, but they take a
little more effort to work properly.  For this reason, they are disabled since
allowing them to be copied and used complicates the workflow for the user.
Until these tests are refactored to be easy to deal with (e.g. does not require
any additional compilation flags, does not require data files).

## SimpleCHull

This test is disabled for a few main reasons:

1. This test requires some defines to work correctly that are defined over
   additional command-line arguments.  This is not ideal and makes using it a
   bit difficult.
2. There is a needed external file: `data/random_input`.  This has two effects:
    1. You cannot run the generated executable in any other directory.  This is
       not desired.
    2. In order to copy litmus tests, you would also need to copy this data
       file.
    3. Makefile dependencies gets a little trickier
3. The test is not self contained in a single source file.  There are three
   other files to use this test: `simple_convex_hull.cpp`,
   `simple_convex_hull.h`, and `s3fp_utils.h`.

If these problems can be aleviated, then this test can come back into the fold.
