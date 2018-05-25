#ifndef EMPTY_BASE_H
#define EMPTY_BASE_H


#include <flit.h>

#include "polybench_utils.hpp"


template <typename T, int M, int N>
class EmptyBase : public flit::TestBase<T> {
public:
  EmptyBase(std::string id) : flit::TestBase<T>(std::move(id)) {}

  virtual size_t getInputsPerRun() override { return N*M; }
  virtual std::vector<T> getDefaultInput() override {
    return random_vector<T>(this->getInputsPerRun());
  }

  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    return vector_string_compare<T>(ground_truth, test_results);
  }

protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<int> sizes = {};
    std::vector<T> C = split_vector(sizes, 0, ti);

    return pickles({});
  }

protected:
  using flit::TestBase<T>::id;
};

#endif  // EMPTY_BASE_H
