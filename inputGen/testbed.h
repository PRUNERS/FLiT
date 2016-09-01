#ifndef TESTBED_H
#define TESTBED_H

#include <string>
#include <functional>
#include <vector>
#include <tuple>

long double
runTestbed(const std::string &testName, const std::vector<float> &inputs);

long double
runTestbed(const std::string &testName, const std::vector<double> &inputs);

long double
runTestbed(const std::string &testName, const std::vector<long double> &inputs);

#endif // TESTBED_H
