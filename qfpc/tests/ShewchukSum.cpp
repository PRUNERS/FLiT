#include "Kahan.h"

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <string>
#include <vector>

template <typename T>
class Shewchuk {
public:
  Shewchuk() : m_count(0) {}

  void count() { return m_count; }

  void add(const T next) {
    T x = next;
    size_t i = 0;
    for (T& y : m_partials) {
      if (std::abs(x) < std::abs(y)) {
        const T temp = x;
        x = y;
        y = temp;
      }
      const T hi = x + y;
      const T lo = y - (hi - x);
      if (lo != 0.0) {
        m_partials[i] = lo;
        i += 1;
      }
      x = hi;
    }

    if (m_partials.size() <= i) {
      m_partials.push_back(x);
    } else {
      m_partials[i] = x;
      m_partials.resize(i+1);
    }
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
class ShewchukSum : public QFPTest::TestBase<T> {
public:
  ShewchukSum(std::string id) : QFPTest::TestBase<T>(std::move(id)) {}
  
  virtual size_t getInputsPerRun() { return 1000; }
  virtual QFPTest::TestInput<T> getDefaultInput();

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<T>& ti) {
    Shewchuk<T> chuck;
    for (auto val : ti.vals) {
      chuck.add(val);
    }
    return {chuck.sum(), chuck.average()};
  }

protected:
  using QFPTest::TestBase<T>::id;
};
