#include "flit.h"

#include <string>

/** An example test class to show how to make FLiT tests
 *
 * You will want to rename this file and rename the class.  Then implement
 * getInputsPerRun(), getDefaultInput() and run_impl().
 */
template <typename T>
class Empty : public flit::TestBase<T> {
public:
  Empty(std::string id) : flit::TestBase<T>(std::move(id)) {}

  /** Specify how many floating-point inputs your algorithm takes.
   * 
   * Can be zero.  If it is zero, then getDefaultInput should return an empty
   * TestInput object which is as simple as "return {};"
   */
  virtual size_t getInputsPerRun() { return 1; }

  /** Specify the default inputs for your test.
   *
   * Used for automated runs.  If you give number in ti.vals than
   * getInputsPerRun, then the run_impl will get called more than once, each
   * time with getInputsPerRun() elements in ti.vals.
   *
   * If your algorithm takes no inputs, then you can simply return an empty
   * TestInput object.  It is as simple as "return {};".
   */
  flit::TestInput<T> getDefaultInput() {
    flit::TestInput<T> ti;
    ti.vals = { 1.0 };
    return ti;
  }

protected:

  /** Call or implement the algorithm here.
   *
   * You need to return two scores, each are of type long double.  Usually the
   * first value is used in analysis and the second score is ignored, so feel
   * free to return 0.0 as the second value if you only need one metric.
   *
   * You are guarenteed that ti will have exactly getInputsPerRun() inputs in
   * it.  If getInputsPerRun() returns zero, then ti.vals will be empty.
   */
  virtual flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
    return {std::pair<long double, long double>(ti.vals[0], 0.0), 0};
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(Empty)
