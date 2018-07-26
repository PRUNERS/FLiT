/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */
#ifndef POLYBENCH_UTILS_H
#define POLYBENCH_UTILS_H


#include <algorithm>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include <cfloat>
#include <cstdint>

/** These REGISTER definitions are helper macro functions that create a
 * specification of the given NAME with different amounts of extra template
 * parameters.
 *
 * These will create test classes named ${NAME}_${DIM0}_${DIM1}_...
 * (for how many extra template parameters exist)
 */
#define POLY_REGISTER_DIM1(NAME, DIM0)                                         \
  template <typename T> class NAME##_##DIM0 : public NAME##Base<T, DIM0> {     \
  public: using NAME##Base<T, DIM0>::NAME##Base;                               \
  };                                                                           \
                                                                               \
  REGISTER_TYPE(NAME##_##DIM0)                                                 \

#define POLY_REGISTER_DIM2(NAME, DIM0, DIM1)                                   \
  template <typename T> class NAME##_##DIM0##_##DIM1                           \
    : public NAME##Base<T, DIM0, DIM1>                                         \
  {                                                                            \
    public: using NAME##Base<T, DIM0, DIM1>::NAME##Base;                       \
  };                                                                           \
                                                                               \
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1)                                        \

#define POLY_REGISTER_DIM3(NAME, DIM0, DIM1, DIM2)                             \
  template <typename T> class NAME##_##DIM0##_##DIM1##_##DIM2                  \
    : public NAME##Base<T, DIM0, DIM1, DIM2>                                   \
  {                                                                            \
    public: using NAME##Base<T, DIM0, DIM1, DIM2>::NAME##Base;                 \
  };                                                                           \
                                                                               \
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2)                               \

#define POLY_REGISTER_DIM4(NAME, DIM0, DIM1, DIM2, DIM3)                       \
  template <typename T> class NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3         \
    : public NAME##Base<T, DIM0, DIM1, DIM2, DIM3>                             \
  {                                                                            \
    public: using NAME##Base<T, DIM0, DIM1, DIM2, DIM3>::NAME##Base;           \
  };                                                                           \
                                                                               \
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3)                      \

#define POLY_REGISTER_DIM5(NAME, DIM0, DIM1, DIM2, DIM3, DIM4)                 \
  template <typename T>                                                        \
  class NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3##_##DIM4                      \
    : public NAME##Base<T, DIM0, DIM1, DIM2, DIM3, DIM4>                       \
  {                                                                            \
    public: using NAME##Base<T, DIM0, DIM1, DIM2, DIM3, DIM4>::NAME##Base;     \
  };                                                                           \
                                                                               \
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3##_##DIM4)             \


/** Generates a random vector of numbers from a seeded random number generator
 *
 * @param n: number of random numbers to generate
 * @param seed: random number generator seed
 * @param lower: lower bound of numbers to generate
 * @param upper: upper bound of numbers to generate
 *
 * @return std::vector containing the generated random numbers
 */
template<typename T>
std::vector<T> seeded_random_vector(size_t n, unsigned int seed, T lower = 0.0,
                                    T upper = std::numeric_limits<T>::max())
{
  std::vector<T> v(n);
  std::minstd_rand g(seed);
  std::uniform_real_distribution<T> dist(lower, upper);
  std::generate_n(v.begin(), n, [&g, &dist]() { return dist(g); });
  return v;
}

/** Generates a random vector with a hard-coded seed
 *
 * This is a convenience function that calls seeded_random_vector().
 */
template<typename T>
std::vector<T> random_vector(size_t n, T lower = 0.0,
                             T upper = std::numeric_limits<T>::max())
{
  return seeded_random_vector<T>(n, 42, lower, upper);
}


/// Compare two strings representing vectors using L1 norm
template<typename T>
long double
vector_string_compare(const std::string &ground_truth,
                      const std::string &test_results) {
  long double absdiff = 0;

  std::stringstream expected(ground_truth);
  std::stringstream actual(test_results);
  T e;
  T a;
  while (expected.good() && actual.good()) {
    expected >> e;
    actual >> a;
    absdiff += std::abs(e - a);
  }

  if (expected.good() != actual.good()) {
    absdiff = std::numeric_limits<long double>::max();
  }

  return absdiff;
}


/** Convert a vector of vectors into a single string
 *
 * Note: the sequence of vectors are collapsed as a single vector
 */
template<typename T>
std::string
pickles(std::initializer_list<std::vector<T>> cucumbers) {
  std::stringstream ss;
  ss << std::setprecision(22); // enough to round trip long doubles

  for (const auto& cuke : cucumbers) {
    for (T c : cuke) {
      ss << c << " ";
    }
  }

  return ss.str();
}


/// Splits the ti vector into multiple disjoint vectors specified by sizes
template<typename T>
std::vector<std::vector<T>>
split_vector(const std::vector<T> &ti, const std::vector<int> &sizes) {
  std::vector<std::vector<T>> split;
  auto start = ti.begin();
  auto stop = start;
  for (size_t i = 0; i < sizes.size(); i++) {
    start = stop;
    stop += sizes[i];
    split.emplace_back(start, stop);
  }
  return split;
}

#endif // POLYBENCH_UTILS_H

