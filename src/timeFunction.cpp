#include "timeFunction.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

namespace flit {

namespace {

int_fast64_t time_function_impl(const TimingFunction &func, size_t loops = 1,
                                size_t repeats = 3)
{
  int_fast64_t min_time = std::numeric_limits<int_fast64_t>::max();
  for (size_t r = 0; r < repeats; r++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < loops; i++) {
      func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    int_fast64_t time = 
        std::chrono::duration_cast<
          std::chrono::duration<int_fast64_t, std::nano>>(end - start).count();
    min_time = std::min(min_time, time);
  }
  return min_time;
}

} // end of unnamed namespace

int_fast64_t time_function(const TimingFunction &func, size_t loops,
                           size_t repeats)
{
  return time_function_impl(func, loops, repeats) / loops;
}

int_fast64_t time_function_autoloop(const TimingFunction &func, size_t repeats)
{
  const size_t min_loops = 1;
  const size_t max_loops = 1e6;
  const int_fast64_t min_time = 2e8; // nanosecs -> 0.2 seconds
  size_t min_avg = SIZE_MAX;
  for (size_t r = 0; r < repeats; r++) {
    size_t time;
    size_t loops;
    for (loops = min_loops; loops <= max_loops; loops *= 10) {
      time = time_function_impl(func, loops, repeats);
      if (time >= min_time) {
        break;
      }
    }
    loops = std::min(max_loops, loops); // in case the for loop reaches the end
    min_avg = std::min(min_avg, time / loops);
  }
  return min_avg;
}

} // end of namespace flit
