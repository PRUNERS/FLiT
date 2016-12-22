#include "Kahan.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <string>
#include <vector>

template <typename T>
class ShewChuck {
public:
  ShewChuck() : m_count(0) {}

  void count() { return m_count; }

  void add(const T next) {
    T x = next;
    size_t partials_len = m_partials.length();
    for (size_t j = 0; j < partials_len; j++) {
      T y = m_partials[j];
      if (std::abs(x) < std::abs(y)) {
        const T temp = x;
        x = y;
        y = temp;
      }
      const T hi = x + y;
      //const T lo = y - ((volatile T)hi - x);
      const T lo = y - (hi - x);
      if (lo != 0.0) {
        m_partials.push_back(lo);
      }
      x = hi;
    }
    m_partials.push_back(x);
    m_count += 1;
  }

  T sum() {
    Kahan<T> summer;
    for (auto val : m_partials) {
      summer.add(val);
    }
    return summer.sum();
  }

  T average() {
    return sum() / m_count;
  }

private:
  std::vector<T> m_partials;
  size_t m_count;
};

template <typename T>
class ShewChuckSum : public QFPTest::TestBase<T> {
public:
  ShewChuckSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}
  
  virtual size_t getInputsPerRun() { return 1000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    ShewChuck<T> chuck;
    for (auto val : ti.vals) {
      chuck.add(val);
    }
    return {chuck.sum(), chuck.average()};
  }

protected:
  using QFPTest::TestBase<T>::id;
};
