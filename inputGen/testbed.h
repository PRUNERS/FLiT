#ifndef TESTBED_H
#define TESTBED_H

#include <string>
#include <functional>
#include <vector>
#include <tuple>

extern "C"
long double
runTestbed_float(const std::string &testName, const std::vector<float> &inputs);

extern "C"
long double
runTestbed_double(const std::string &testName, const std::vector<double> &inputs);

extern "C"
long double
runTestbed_longdouble(const std::string &testName, const std::vector<long double> &inputs);

#endif // TESTBED_H
