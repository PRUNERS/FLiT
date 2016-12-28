#ifndef SHEWCHUK_H
#define SHEWCHUK_H

#include "Kahan.h"

#include <complex>    // std::abs
#include <vector>

#include <cstddef>    // size_t

template <typename T>
class Shewchuk {
public:
  Shewchuk() : m_count(0), m_cachedSum(0), m_isCachedSumValid(true) {}

  void count() { return m_count; }

  void add(const T next) {
    m_isCachedSumValid = false; // invalidate the cache
    T x = next;
    auto n = m_partials.size();
		decltype(n) i = 0;
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

    if (n > i) {
      m_partials.resize(i);
    }
    if (x != 0.0) {
      m_partials.push_back(x);
    }
    m_count += 1;
  }

  T sum() {
    if (!m_isCachedSumValid) {
      Kahan<T> summer;
      for (auto val : m_partials) {
        summer.add(val);
      }
      m_cachedSum = summer.sum();
      m_isCachedSumValid = true;
    }
    return m_cachedSum;
  }

	// Note: this does not set or return the cache.  Only sum() does that.
  T sum2() {
    T x, y, lo, yr, hi = 0.0; 
		auto p = m_partials;
		auto n = p.size();

		// This code was copied verbatim from Python's math.fsum() implementation
    if (n > 0) { 
      hi = p[--n];
      /* sum_exact(ps, hi) from the top, stop when the sum becomes
         inexact. */
      while (n > 0) { 
        x = hi;
        y = p[--n];
        //assert(fabs(y) < fabs(x));
        hi = x + y; 
        yr = hi - x; 
        lo = y - yr;
        if (lo != 0.0) 
          break;
      }
      /* Make half-even rounding work across multiple partials.
         Needed so that sum([1e-16, 1, 1e16]) will round-up the last
         digit to two instead of down to zero (the 1e-16 makes the 1
         slightly closer to two).  With a potential 1 ULP rounding
         error fixed-up, math.fsum() can guarantee commutativity. */
      if (n > 0 && ((lo < 0.0 && p[n-1] < 0.0) ||
                    (lo > 0.0 && p[n-1] > 0.0))) {
        y = lo * 2.0; 
        x = hi + y; 
        yr = x - hi;
        if (y == yr) {
          hi = x;
				}
      }
    }

		return hi;
  }

  T average() {
    return sum() / m_count;
  }

  std::vector<T>& partials() { return m_partials; }

private:
  std::vector<T> m_partials;
  size_t m_count;
  T m_cachedSum;
  bool m_isCachedSumValid;
};

#endif // SHEWCHUK_H
