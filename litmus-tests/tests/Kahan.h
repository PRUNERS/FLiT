#ifndef KAHAN_H
#define KAHAN_H

template <typename T>
class Kahan {
public:
  Kahan() : m_sum(0), m_c(0) {}

  void add(const T next) {
    const T y = next - m_c;
    const T t = m_sum + y;
    m_c = (t - m_sum) - y;
    m_sum = t;
  }

  T sum() { return m_sum; }

private:
  T m_sum;
  T m_c;
};

#endif // KAHAN_H
