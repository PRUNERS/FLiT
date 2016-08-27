#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <vector>

template <typename T>
inline T distribution(const std::vector<T> &in)
{
  T a = in[0];
  T b = in[1];
  T c = in[2];
  return (a * c) + (b * c);
}

#endif // FUNCTIONS_HPP
