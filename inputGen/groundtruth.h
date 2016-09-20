#ifndef GROUNDTRUTH_H
#define GROUNDTRUTH_H

#include <string>
#include <functional>
#include <vector>
#include <tuple>

template <typename T>
using TruthType = std::pair<std::vector<T>, long double>;

TruthType<float>
runGroundtruth(const std::string &testName, std::function<float()> randGen);

TruthType<double>
runGroundtruth(const std::string &testName, std::function<double()> randGen);

TruthType<long double>
runGroundtruth(const std::string &testName, std::function<long double()> randGen);

#endif // GROUNDTRUTH_H
