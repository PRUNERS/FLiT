#include <flit.h>

#include <string>
#include <random>

using namespace std;

template <typename T, typename Gen, typename Dist>
class Random : public flit::TestBase<T> {
public:
  Random(string id) : flit::TestBase<T>(move(id)) {}
  virtual size_t getInputsPerRun() override { return 1; }
  virtual vector<T> getDefaultInput() override { return { 42, 24, 12, 10, 103 }; }
protected:
  virtual flit::Variant run_impl(const vector<T> &ti) override {
    size_t seed = ti[0];
    Gen g(seed);
    return Dist()(g);
  }
protected:
  using flit::TestBase<T>::id;
};

template <typename Gen>
struct Pass {
  auto operator() (Gen &g) -> decltype(g()) {
    return g();
  }
};

template <typename T>
struct Canonical {
  template <typename G>
  T operator() (G &g) {
    return generate_canonical<T, numeric_limits<T>::digits>(g);
  }
};

#define MY_REGISTRATION(name, gen, dist) \
  template <typename T> \
  class name : public Random<T, gen, dist> { \
    using Random<T, gen, dist>::Random; \
  }; \
  REGISTER_TYPE(name)

#define REGISTER_GENERATOR(name, klass) \
  MY_REGISTRATION(Random_##name##_Pass,              klass, Pass<klass>) \
  MY_REGISTRATION(Random_##name##_uniformint,        klass, uniform_int_distribution<long>) \
  MY_REGISTRATION(Random_##name##_uniformreal,       klass, uniform_real_distribution<T>) \
  MY_REGISTRATION(Random_##name##_binomial,          klass, binomial_distribution<int>) \
  MY_REGISTRATION(Random_##name##_bernoulli,         klass, bernoulli_distribution) \
  MY_REGISTRATION(Random_##name##_geometric,         klass, geometric_distribution<int>) \
  MY_REGISTRATION(Random_##name##_negative_binomial, klass, negative_binomial_distribution<int>) \
  MY_REGISTRATION(Random_##name##_poisson,           klass, poisson_distribution<int>) \
  MY_REGISTRATION(Random_##name##_exponential,       klass, exponential_distribution<T>) \
  MY_REGISTRATION(Random_##name##_gamma,             klass, gamma_distribution<T>) \
  MY_REGISTRATION(Random_##name##_weibull,           klass, weibull_distribution<T>) \
  MY_REGISTRATION(Random_##name##_extreme_value,     klass, extreme_value_distribution<T>) \
  MY_REGISTRATION(Random_##name##_normal,            klass, normal_distribution<T>) \
  MY_REGISTRATION(Random_##name##_lognormal,         klass, lognormal_distribution<T>) \
  MY_REGISTRATION(Random_##name##_chi_squared,       klass, chi_squared_distribution<T>) \
  MY_REGISTRATION(Random_##name##_cauchy,            klass, cauchy_distribution<T>) \
  MY_REGISTRATION(Random_##name##_fisher_f,          klass, fisher_f_distribution<T>) \
  MY_REGISTRATION(Random_##name##_student_t,         klass, student_t_distribution<T>) \
  MY_REGISTRATION(Random_##name##_canonical,         klass, Canonical<T>)

REGISTER_GENERATOR(mt19937, mt19937)
REGISTER_GENERATOR(mt19937_64, mt19937_64)
REGISTER_GENERATOR(default, default_random_engine);
REGISTER_GENERATOR(minstd_rand, minstd_rand);
REGISTER_GENERATOR(minstd_rand0, minstd_rand0);
REGISTER_GENERATOR(ranlux24, ranlux24);
REGISTER_GENERATOR(ranlux48, ranlux48);
REGISTER_GENERATOR(knuth_b, knuth_b);


