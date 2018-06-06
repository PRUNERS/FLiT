#include <flit.h>

#include <string>
#include <random>

template <typename T, typename Gen, typename Dist>
class Random : public flit::TestBase<T> {
public:
  Random(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 1; }
  virtual std::vector<T> getDefaultInput() override { return { 42, 24, 12, 10, 103 }; }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    size_t seed = ti[0];
    Gen g(seed);
    return Dist()(g);
  }
protected:
  using flit::TestBase<T>::id;
};

#define MY_REGISTRATION(name, gen, dist) \
  template <typename T> \
  class name : public Random<T, gen, dist> { \
    using Random<T, gen, dist>::Random; \
  }; \
  REGISTER_TYPE(name)

MY_REGISTRATION(Random_mt19937_uniform,           std::mt19937, std::uniform_real_distribution<T>)
MY_REGISTRATION(Random_mt19937_binomial,          std::mt19937, std::binomial_distribution<int>)
MY_REGISTRATION(Random_mt19937_bernoulli,         std::mt19937, std::bernoulli_distribution)
MY_REGISTRATION(Random_mt19937_geometric,         std::mt19937, std::geometric_distribution<int>)
MY_REGISTRATION(Random_mt19937_negative_binomial, std::mt19937, std::negative_binomial_distribution<int>)
MY_REGISTRATION(Random_mt19937_poisson,           std::mt19937, std::poisson_distribution<int>)
MY_REGISTRATION(Random_mt19937_exponential,       std::mt19937, std::exponential_distribution<T>)
MY_REGISTRATION(Random_mt19937_gamma,             std::mt19937, std::gamma_distribution<T>)
MY_REGISTRATION(Random_mt19937_weibull,           std::mt19937, std::weibull_distribution<T>)
MY_REGISTRATION(Random_mt19937_extreme_value,     std::mt19937, std::extreme_value_distribution<T>)
MY_REGISTRATION(Random_mt19937_normal,            std::mt19937, std::normal_distribution<T>)
MY_REGISTRATION(Random_mt19937_lognormal,         std::mt19937, std::lognormal_distribution<T>)
MY_REGISTRATION(Random_mt19937_chi_squared,       std::mt19937, std::chi_squared_distribution<T>)
MY_REGISTRATION(Random_mt19937_cauchy,            std::mt19937, std::cauchy_distribution<T>)
MY_REGISTRATION(Random_mt19937_fisher_f,          std::mt19937, std::fisher_f_distribution<T>)
MY_REGISTRATION(Random_mt19937_student_t,         std::mt19937, std::student_t_distribution<T>)
MY_REGISTRATION(Random_mt19937_discrete,          std::mt19937, std::discrete_distribution<T>)


