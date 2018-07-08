# Writing Test Cases

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](mpi-support.md)

When you initialize a test directory using `flit init`, a test file called
`Empty.cpp` will be placed in the `tests` directory.  This test is as the name
implies - empty.  It basically shows you how to create a new test and is
intended to be used as a template for creating tests.

To add a test, simply copy `Empty.cpp` and place it in the `tests` directory
with an ending of `cpp`.  Rename the class inside, and you're good to go.


## Test Case Requirements

The tests you write in the FLiT framework are different from unit tests or even
verification tests.  We will call them **reproducibility tests** or **flit
tests**.  One key difference is that, unlike unit tests, a golden value does
not need to be known a priori.  Instead, the return value from the ground-truth
compilation will be used as the base comparison value.

This is also covered by the comments within the `Empty.cpp` file given to you.
Each test class has the following pure virtual functions (which means functions
you must implement for it to compile):

- `getInputsPerRun()`: This function simply returns the number of
  floating-point inputs your test takes.
- `getDefaultInput()`: This function returns the default input your test takes,
  in the form of a `std::vector<T>`.  If you want different inputs for `float`,
  `double`, and `long double`, give a template specialization of this function
  for each of those precisions.  You can also use this function to provide
  data-driven tests.  If your test takes 3 inputs, and this function returns 9
  values, then the test will be run 3 times, once for each set of inputs.
- `run_impl()`: This is where the actual test lives.  It receives a vector of
  floating-point values as input that will be exactly as long as what
  `getInputsPerRun()` returns.  The test returns a `flit::Variant` object,
  which of the time of this writing can handle `long double` and `std::string`.

There are some optional functions that you can override as well.  The default
implementation is provided for you in `Empty.cpp` so you can see if you would
like to override that behavior.

- `compare(long double, long double)`: This is a custom comparison value that
  is used to compare the results from the ground-truth compilation and all
  other compilations.  If your test returns a floating-point value, then this
  compare function is the one that is used.  Common examples of comparisons
  here are absolute error and relative error.
- `compare(string, string)`: This is a custom comparison value that is used to
  compare the results from the ground-truth compilation and all other
  compilations.  If your test returns a `std::string`, then this compare
  function is the one that is used.  There was no good default, so it is highly
  recommended to implement your own.  For example, your strings may represent a
  grid of values, in which case you could convert them back to grids and
  perform an $\ell_2$ norm between the two grids.


## Disabling Test Cases

Since all files ending with `.cpp` in the `tests` directory and the parent
directory are included in the compilation, the best way to disable a test is
simply to move it into a new directory.  Suppose the test I want to disable is
called `BrokenTest.cpp`.

```bash
mkdir disabled
mv tests/BrokenTest.cpp disabled
```

That's it.


## Only Using Particular Precisions

You may have code you want tested that only has implemented one precision, for
example `double` precision.  That's okay, you can still use flit for your code
without having lots of trouble.

In the class declaration of the templated test class of yours, simply have an
empty `run_impl()` function like the following:

```c++
// Default implementation does nothing
virtual flit::Variant run_impl(const std::vector<T>& ti) override {
  FLIT_UNUSED(ti);
  return flit::Variant();
}
```

If an empty `flit::Variant()` is returned from `run_impl()`, then that test is
considered disabled.  The beauty is that you can specialize this and only
implement it for a particular precision.  After the template class declaration,
you can now implement only the `double` precision version to return something
meaningful:

```c++
template<>
flit::Variant MyTestClass<double>::run_impl(const std::vector<double>& ti) {
   // test logic here ...
   return something_meaningful;
}
```

Using this approach, you can have this test only implemented for `double`
precision, and have the other precisions disabled.  That way, you will not have
misleading information of your test case using `float` or `long double`
precision taking up space in your results database.


## Writing Tests With More Template Parameters

There may be times when you want to write tests that have more than one
template parameter.  For example, you may want an integer template parameter
that specifies the size of the problem to solve.  This section will guide you
to be able to leverage templates to create many tests in a few number of lines
of code.

As an example, let us take the example of code that takes the dot product, but with different sizes of vectors.

```c++
#include <flit.h>

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cmath>

// Define a highly templatized class to make many tests
template <typename T, int Len>
class DotProduct : public flit::TestBase<T> {
public:
  DotProduct(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return Len * 2; }
  virtual std::vector<T> getDefaultInput() override {
    // Could come up with a better mechanism to generate inputs
    std::vector<T> inputs(getInputsPerRun());
    size_t seed = 42;
    std::minstd_rand engine(seed);
    std::uniform_real_distribution<T> dist(
        T(), std::sqrt(std::numeric_limits<T>::max()) / 2);
    for (T &x : inputs) {
      x = dist(engine);
    }
    return inputs;
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    std::vector<T> a(ti.begin(), ti.begin() + Len);
    std::vector<T> b(ti.begin() + Len, ti.end());
    long double val{0.0};
    for (int i = 0; i < Len; i++) {
      val += a[i] * b[i];
    }
    return val;
  }
};

#define DOT_PRODUCT_REGISTRATION(len) \
  template <typename T> \
  class DotProduct_Len##len : public DotProduct<T, len> { \
    using DotProduct<T, len>::DotProduct; \
  }; \
  REGISTER_TYPE(DotProduct_Len##len)

// Create 8 tests with different vector sizes
DOT_PRODUCT_REGISTRATION(3)
DOT_PRODUCT_REGISTRATION(4)
DOT_PRODUCT_REGISTRATION(5)
DOT_PRODUCT_REGISTRATION(6)
DOT_PRODUCT_REGISTRATION(10)
DOT_PRODUCT_REGISTRATION(100)
DOT_PRODUCT_REGISTRATION(1000)
DOT_PRODUCT_REGISTRATION(10000)
```


[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](mpi-support.md)
