#include <flit.h>

#include <string>

template <typename T>
GLOBAL
void Empty_kernel(const T* const* tiList, double* results) {
#ifdef __CUDA__
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
  auto idx = 0;
#endif
  auto& ti = tiList[idx];
  results[idx] = ti.vals[0];
}

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
   * std::vector, which is as simple as "return {};"
   */
  virtual size_t getInputsPerRun() override { return 1; }

  /** Specify the default inputs for your test.
   *
   * Used for automated runs.  If you give number in ti.vals than
   * getInputsPerRun, then the run_impl will get called more than once, each
   * time with getInputsPerRun() elements in ti.vals.
   *
   * If your algorithm takes no inputs, then you can simply return an empty
   * std::vector object.  It is as simple as "return {};".
   */
  virtual std::vector<T> getDefaultInput() override {
    return { 1.0 };
  }

  /** Custom comparison methods
   *
   * These comparison operations are meant to create a metric between the test
   * results from this test in the current compilation, and the results from
   * the ground truth compilation.  You can do things like the relative error
   * or the absolute error (for the case of long double).
   *
   * The below specified functions are the default implementations defined in
   * the base class.  It is safe to delete these two functions if this
   * implementation is adequate for you.
   *
   * Which one is used depends on the type of Variant that is returned from the
   * run_impl function.  The value returned by compare will be the value stored
   * in the database for later analysis.
   *
   * Note: when using the CUDA kernel functionality, only long double return
   * values are valid for now.
   */
  virtual long double compare(long double ground_truth,
                              long double test_results) const override {
    // absolute error
    return test_results - ground_truth;
  }

  /** There is no good default implementation comparing two strings */
  virtual long double compare(const std::string &ground_truth,
                              const std::string &test_results) const override {
    FLIT_UNUSED(ground_truth);
    FLIT_UNUSED(test_results);
    return 0.0;
  }

protected:
  /** Return a kernel pointer to the CUDA kernel equivalent of run_impl
   *
   * The default base implementation of this function is simply to return
   * null_ptr, which will cause it to call run_impl even when compiled under
   * CUDA.
   *
   * If you do not have or do not want to have a CUDA version of your code,
   * then you can delete this virtual function and use the base implementation.
   *
   * See the documentation above Empty_kernel() for details about what the
   * kernel is expected to have.
   */
  virtual flit::KernelFunction<T>* getKernel() override { return Empty_kernel; }

  /** Call or implement the algorithm here.
   *
   * You return a flit::Variant which can represent one of a number of
   * different types.  See the flit::Variant class for more details.  In short,
   * it can represent a long double, a std::string, or an empty nothing (at
   * least as of this writing.  See the flit::Variant documentation for more
   * up-to-date information).  If you return an empty flit::Variant, then that
   * result will not show up in the output (meaning that test is disabled).
   * This is desirable for example if you only have functionality with double
   * precision and want to disable the float and long double test
   * implementations (simply have those return an empty flit::Variant).
   *
   * The value returned by run_impl is the same value used in compare()
   * implemented above.
   */
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    return flit::Variant();
  }

protected:
  using flit::TestBase<T>::id;
};

REGISTER_TYPE(Empty)
