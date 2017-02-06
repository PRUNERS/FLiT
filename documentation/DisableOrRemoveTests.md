#Disabling or Removing Litmus Tests
The simplest way to remove tests from the next run is to (re)move the source code from the directory `QFP/qfpc/tests`

	1. cd QFP/qfpc
	2. mkdir disabled_tests
	3. mv <tests_to_disable> disabled_tests

If you want to compile a single test on the current machine for debug purposes then run:

	1. cd QFP/qfpc
	2. make -f MakefileDev TESTS=<test_cpp_file>

