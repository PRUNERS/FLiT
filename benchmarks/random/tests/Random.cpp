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

#include <flit.h>

#include <string>
#include <random>

using namespace std;

template <typename T, typename Gen, typename Dist>
class Random : public flit::TestBase<T> {
public:
  Random(string id) : flit::TestBase<T>(move(id)) {}
  virtual size_t getInputsPerRun() override { return 1; }
  virtual vector<T> getDefaultInput() override { return { 0, 42, 24, 12, 103 }; }
protected:
  virtual flit::Variant run_impl(const vector<T> &ti) override {
    size_t seed = ti[0];
    Gen g(seed);
    return Dist()(g);
  }
protected:
  using flit::TestBase<T>::id;
};

// A simple passthrough distribution that returns the generator's value
template <typename Gen>
struct Pass {
  auto operator() (Gen &g) -> decltype(g()) {
    return g();
  }
};

// A distribution between [0, 1) using std::generate_canonical()
template <typename T>
struct Canonical {
  template <typename G>
  T operator() (G &g) {
    return generate_canonical<T, numeric_limits<T>::digits>(g);
  }
};

// Convenience macro to create new tests with a generator and distribution
#define MY_REGISTRATION(name, gen, dist) \
  template <typename T> \
  class name : public Random<T, gen, dist> { \
    using Random<T, gen, dist>::Random; \
  }; \
  REGISTER_TYPE(name)

// Convenience macro to create a suite of tests varying the distribution for the given generator
#define REGISTER_GENERATOR(name, klass) \
  MY_REGISTRATION(Random_##name##_Pass,              klass, Pass<klass>                        ) \
  MY_REGISTRATION(Random_##name##_uniformint,        klass, uniform_int_distribution<long>     ) \
  MY_REGISTRATION(Random_##name##_uniformreal,       klass, uniform_real_distribution<T>       ) \
  MY_REGISTRATION(Random_##name##_binomial,          klass, binomial_distribution<int>         ) \
  MY_REGISTRATION(Random_##name##_bernoulli,         klass, bernoulli_distribution             ) \
  MY_REGISTRATION(Random_##name##_geometric,         klass, geometric_distribution<int>        ) \
  MY_REGISTRATION(Random_##name##_negative_binomial, klass, negative_binomial_distribution<int>) \
  MY_REGISTRATION(Random_##name##_poisson,           klass, poisson_distribution<int>          ) \
  MY_REGISTRATION(Random_##name##_exponential,       klass, exponential_distribution<T>        ) \
  MY_REGISTRATION(Random_##name##_gamma,             klass, gamma_distribution<T>              ) \
  MY_REGISTRATION(Random_##name##_weibull,           klass, weibull_distribution<T>            ) \
  MY_REGISTRATION(Random_##name##_extreme_value,     klass, extreme_value_distribution<T>      ) \
  MY_REGISTRATION(Random_##name##_normal,            klass, normal_distribution<T>             ) \
  MY_REGISTRATION(Random_##name##_lognormal,         klass, lognormal_distribution<T>          ) \
  MY_REGISTRATION(Random_##name##_chi_squared,       klass, chi_squared_distribution<T>        ) \
  MY_REGISTRATION(Random_##name##_cauchy,            klass, cauchy_distribution<T>             ) \
  MY_REGISTRATION(Random_##name##_fisher_f,          klass, fisher_f_distribution<T>           ) \
  MY_REGISTRATION(Random_##name##_student_t,         klass, student_t_distribution<T>          ) \
  MY_REGISTRATION(Random_##name##_canonical,         klass, Canonical<T>                       )

// Create all of the tests now for various generators
REGISTER_GENERATOR(mt19937,      mt19937              )
REGISTER_GENERATOR(mt19937_64,   mt19937_64           )
REGISTER_GENERATOR(default,      default_random_engine)
REGISTER_GENERATOR(minstd_rand,  minstd_rand          )
REGISTER_GENERATOR(minstd_rand0, minstd_rand0         )
REGISTER_GENERATOR(ranlux24,     ranlux24             )
REGISTER_GENERATOR(ranlux48,     ranlux48             )
REGISTER_GENERATOR(knuth_b,      knuth_b              )


