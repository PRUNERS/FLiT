# Tests for FLiT

Yes, I know this is a testing framework.  But, how can you trust your testing
framework unless it is itself tested?  This directory contains regression tests
for the FLiT framework.  We may not have full coverage, but it gives us some
confidence that regressions do not occur.

As part of this suite of regression tests, there is a test harness in
`test_harness.h`.  This is intended to be a very simple C++ testing framework
so that we can test the C++ functionality of FLiT.  To make sure this new
harness works correctly, there is also a tester for the harness called
`harness_tester.cpp`.  This is part of the set of tests that are executed.

I know, we have a minimal framework for testing our test framework, and then
simple tests to test the minimal framework that will be used to test our test
framework.  Believe me, it's worth it to be confident that your code works as
expected.

To run all of the tests, simply type `make check` from either this directory or
from the top-level directory.


