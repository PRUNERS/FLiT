#ifndef POLYBENCH_UTILS_H
#define POLYBENCH_UTILS_H


#include <cstdint>
#include <iomanip>
#include <sstream>


#define REGISTER_1(NAME, DIM0)						\
  template <typename T> class NAME##_##DIM0 : public NAME##Base<T, DIM0> { \
  public: using NAME##Base<T, DIM0>::NAME##Base;			\
  };									\
  									\
  REGISTER_TYPE(NAME##_##DIM0)						\

#define REGISTER_2(NAME, DIM0, DIM1)					\
  template <typename T> class NAME##_##DIM0##_##DIM1 : public NAME##Base<T, DIM0, DIM1> { \
  public: using NAME##Base<T, DIM0, DIM1>::NAME##Base;			\
  };									\
  									\
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1)					\

#define REGISTER_3(NAME, DIM0, DIM1, DIM2)				\
  template <typename T> class NAME##_##DIM0##_##DIM1##_##DIM2 : public NAME##Base<T, DIM0, DIM1, DIM2> { \
  public: using NAME##Base<T, DIM0, DIM1, DIM2>::NAME##Base;		\
  };									\
  									\
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2)			\

#define REGISTER_4(NAME, DIM0, DIM1, DIM2, DIM3)			\
  template <typename T> class NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3 : public NAME##Base<T, DIM0, DIM1, DIM2, DIM3> { \
  public: using NAME##Base<T, DIM0, DIM1, DIM2, DIM3>::NAME##Base;	\
  };									\
  									\
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3)		\

#define REGISTER_5(NAME, DIM0, DIM1, DIM2, DIM3, DIM4)			\
  template <typename T> class NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3##_##DIM4 : public NAME##Base<T, DIM0, DIM1, DIM2, DIM3, DIM4> { \
  public: using NAME##Base<T, DIM0, DIM1, DIM2, DIM3, DIM4>::NAME##Base; \
  };									\
  									\
  REGISTER_TYPE(NAME##_##DIM0##_##DIM1##_##DIM2##_##DIM3##_##DIM4)	\


template<typename T> inline T max_f();
template<> inline float max_f<float>() { return FLT_MAX; }
template<> inline double max_f<double>() { return DBL_MAX; }
template<> inline long double max_f<long double>() { return LDBL_MAX; }


template<typename T>
std::vector<T> seeded_random_vector(size_t n, unsigned int seed) {
  std::vector<T> v(n);
  srand(seed);

  // IB: Not a good rng algo, improve later
  uint32_t pad = 0;
  for (int i = 0; i < n; i++) {
    if ((i)%31 == 0) {
      pad = static_cast<uint32_t>(rand());
    }
    int sign = -1 * (pad & 0x1);
    pad >>= 1;

    v[i] = sign * static_cast<double>(rand()) / RAND_MAX * max_f<T>();
  }
  return v;
}


template<typename T>
std::vector<T> random_vector(size_t n) {
  return seeded_random_vector<T>(n, 42);
}



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
    absdiff = max_f<long double>();
  }

  return absdiff;
}


template<typename T>
std::string
pickles(std::initializer_list<std::vector<T>> cucumbers) {
  std::stringstream ss;
  ss << std::setprecision(22); // enough to round trip long doubles

  for (std::vector<T> cuke : cucumbers) {
    for (T c : cuke) {
      ss << c << " ";
    }
  }

  return ss.str();
}

template<typename T>
std::vector<T>
split_vector(std::vector<int> sizes, int index, std::vector<T> ti) {
  int start = 0;
  int end = 0;
  int i = 0;
  for (; i<index; i++) {
    start += sizes[i];
  }
  end = start + sizes[i];
  return std::vector<T>(ti.begin() + start, ti.begin() + end);
}

#endif // POLYBENCH_UTILS_H

