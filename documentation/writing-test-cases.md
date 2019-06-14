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
with an ending of `.cpp`.  Rename the class inside, and you're good to go.  If
you name it with a different ending than `.cpp`, then you will need to ensure
it is included from within `custom.mk` in the `SOURCE` variable.


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
  which of the time of this writing can handle `long double`, `std::string`,
  `std::vector<std::string>`, and `std::vector<T>`, where T is the
  floating-point type of the templated test class instance.

There are some optional functions that you can override as well.  The default
implementation is provided for you in `Empty.cpp` so you can see if you would
like to override that behavior.

- `compare()`: This is a custom comparison value that is used to compare the
  results from the ground-truth compilation and all other compilations.  There
  are multiple versions of this function.  Override the correct one for the
  value returned from `run_impl()`.
    - `compare(long double, long double)`: used if your test returns a
      floating-point value.  The default implementation is to calculate the
      absolute error.
    - `compare(string, string)`: used if your test returns a string.  There is
      no good default implementation, so be sure to override this if used.
    - `compare(std::vector<string>, std::vector<string>)`: used if your test
      returns a vector of strings.  There is no good default implementation, so
      be sure to override this if used.
    - `compare(std::vector<T>, std::vector<T>)`: used if your test returns a
      vector of floating-point values.  The default implementation calculates
      the L2 norm (using `flit::l2norm()`), which is great if that is what you
      want.  Otherwise, you will want to override this function.


## Disabling Test Cases

Since all files ending with `.cpp` in the `tests` directory and the parent
directory are included in the compilation by default, the best way to disable a
test is simply to move it into a new directory.  Suppose the test I want to
disable is called `BrokenTest.cpp`.

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


## Wrapping Around Main Function

Often times, we want to test our entire application with FLiT.  In that case,
you may want to wrap around the `main()` function of all of your code.  Well,
you would be in luck.  FLiT provides a means and a pattern to wrap around your
`main()` function.  Here is an example of such a use case:

```c++
// rename main() to mpiapp_main() to not cause linker problems
#define main myapp_main
#include "myapp.cpp"
#undef main

// this allows flit to use this in call_main() and call_mpi_main()
FLIT_REGISTER_MAIN(myapp_main);

#include "flit.h"

template <typename T>
class MyAppTest : public flit::TestBase<T> {
public:
  MyAppTest(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual long double compare(const std::vector<std::string> &baseline,
                              const std::vector<std::string> &results)
  const override {
    // your custom comparison implementation
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }
protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(MyAppTest);

template<>
flit::Variant MyAppTest<float> run_impl(const std::vector<float> &ti) {
  FLIT_UNUSED(ti);
  auto results = flit::call_main(myapp_main, "myapp", "--iters 20 -v");
  std::vector<std::string> vec_results;
  if (results.ret != 0) {
    throw std::logic_error("mpi_app failed in its execution");
  }
  vec_results.push_back(results.out);
  vec_results.push_back(results.err);
  return vec_results;
}
```

In the above example, we only use the float precision variant of the test
class
(see [Only Using Particular Precisions](#only-using-particular-precisions)]).
Also, the `main()` function is being called in a separate process using the
function `flit::call_mpi_main()`.

Let's step through this example:

```c++
// rename main() to mpiapp_main() to not cause linker problems
#define main myapp_main
#include "myapp.cpp"
#undef main

// this allows flit to use this in call_main() and call_mpi_main()
FLIT_REGISTER_MAIN(myapp_main);
```

This is how to wrap around a `main()` function using FLiT.

1. Exclude the source file containing the `main()` function from FLiT test
   compilation.
2. Include that source file in your test class, but before doing so, rename
   `main` to something else so that we can use the `main()` function provided
   by the FLiT testing framework.
3. Register this newly named `main()` function (in this example, it was renamed
   to `myapp_main()`) using `FLIT_REGISTER_MAIN()`.  This macro must be placed
   in the global scope (it also works from within a namespace).

Once you do these three steps, you are allowed to then use `flit::call_main()`.

```c++
#include "flit.h"

template <typename T>
class MyAppTest : public flit::TestBase<T> {
public:
  MyAppTest(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual long double compare(const std::vector<std::string> &baseline,
                              const std::vector<std::string> &results)
  const override {
    // your custom comparison implementation
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }
protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(MyAppTest);
```

The above you might recognize as a simple empty test with a
`std::vector<std::string>` comparison override.  Nothing new here.

```c++
template<>
flit::Variant MyAppTest<float> run_impl(const std::vector<float> &ti) {
  FLIT_UNUSED(ti);
  auto results = flit::call_main(myapp_main, "myapp", "--iters 20 -v");
  std::vector<std::string> vec_results;
  if (results.ret != 0) {
    throw std::logic_error("mpi_app failed in its execution");
  }
  vec_results.push_back(results.out);
  vec_results.push_back(results.err);
  return vec_results;
}
```

Above, we implement only the `float` variant of the `run_impl()` function.  Within, we call `flit::call_main()` with the following arguments:

1. The function pointer that follows the same signature as a `main()` function
   (i.e., returns `int` and takes an `int` and a `char**`).  In this case, the
   pointer to the `myapp_main()` function.
2. The name you want to be used as the executable name when that `myapp_main()`
   function is called.  The value of this argument will be placed in `arg[0]`.
   In this example, we want the `arg[0]` parameter to be `"myapp"`.  Some
   applications behave differently depending on the name of the executable
   being run.  For example, `bash` behaves differently if it is called from a
   symbolic link called `sh`.
3. The rest of the command-line parameters are given.  This is a single string
   to encode the rest of the parameters.  This is to be formed in a way that it
   would work when calling your application from the shell.  That means
   escaping certain characters that would be interpretted by the shell.  Here,
   you can use pipes, semicolons and other tricks, but tread carefully as some
   unexpected behavior may follow.

The function `flit::call_main()` actually creates a child process of the
current test executable (a.k.a., recursion), but that child process shortcuts
to calling the provided function pointer and returning.  Since it is in a child
process, you are safe to use standard out, standard error, and even functions
like `exit()`.

The value returned from `flit::call_main()` is a struct called `flit::ProcResult`.

```c++
namespace flit {

struct ProcResult {
  int ret;
  std::string out;
  std::string err;
}

}
```

You are given the standard output and standard error produced by your given
function as well as the return code of the child process.  Feel free to use
these to convey your answer back to the test in order to return a value for
later comparison.  Alternatively, you can use output files from your `main()`
function to communicate the results of the computations.

**Note:** There is no current way using FLiT functions to interact with the
child process.  By that I mean you cannot provide standard input.  If that is
desired, please issue a GitHub issue.


## Writing MPI Tests

Most cases we have seen for MPI code to be tested has been when a `main()`
function was to be wrapped in a FLiT test.  Suppose then that we have a
`main()` function in a file call `mpi_app.cpp`, and we want it called from our
test called `MpiAppTest`.  The test may look something like the following:

```c++
// rename main() to mpiapp_main() to not cause linker problems
#define main mpiapp_main
#include "mpi_app.cpp"
#undef main

// this allows flit to use this in call_main() and call_mpi_main()
FLIT_REGISTER_MAIN(mpiapp_main);

#include "flit.h"

template <typename T>
class MpiAppTest : public flit::TestBase<T> {
public:
  MpiAppTest(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual long double compare(const std::vector<std::string> &baseline,
                              const std::vector<std::string> &results)
  const override {
    // your custom comparison implementation
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }
protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(MpiAppTest);

template<>
flit::Variant MpiAppTest<double> run_impl(const std::vector<double> &ti) {
  FLIT_UNUSED(ti);
  auto results = flit::call_mpi_main(
      mpiapp_main, "mpirun -n 4", "mpi_app", "--data ../data/star.dat");
  std::vector<std::string> vec_results;
  if (results.ret != 0) {
    throw std::logic_error("mpi_app failed in its execution");
  }
  vec_results.push_back(results.out);
  vec_results.push_back(results.err);
  return vec_results;
}
```

In the above example, we only use the double precision variant of the test
class
(see [Only Using Particular Precisions](#only-using-particular-precisions)]).
Also, the `main()` function is being called in a separate process (see
[Wrapping Around Main Function](#wrapping-around-main-function)) with four
processes, because of the `"mpirun -n 4"` argument passed to
`flit::call_mpi_main()`.

Let's step through this example:

```c++
// rename main() to mpiapp_main() to not cause linker problems
#define main mpiapp_main
#include "mpi_app.cpp"
#undef main

// this allows flit to use this in call_main() and call_mpi_main()
FLIT_REGISTER_MAIN(mpiapp_main);
```

Similarly to how you would use `call_main()`, you do the same for any MPI test.  This is done for many reasons:

- it is safe to call `MPI_Init()` and `MPI_Finalize()` in this way as a child
  process.  Otherwise, when would and could they be called?
- user code will be executed as expected, no surprises
- calls such as `terminate()`, `abort()`, `exit()`, or even `MPI_abort()` are
  totally fine to do, since it happens in a child process
- each test can use a different MPI run configuration (e.g., one test can run
  with 2 MPI processes and another can run with 4 MPI processes).
- we can safely run the MPI code more than once to get reliable timing

But that does mean that all MPI tests will need to have an entry point from a
main-like function and will have to call `MPI_Init()` and `MPI_Finalize()`
accorddingly.

```c++
#include "flit.h"

template <typename T>
class MpiAppTest : public flit::TestBase<T> {
public:
  MpiAppTest(std::string id) : flit::TestBase<T>(std::move(id)) {}
  virtual size_t getInputsPerRun() override { return 0; }
  virtual std::vector<T> getDefaultInput() override { return {}; }
  virtual long double compare(const std::vector<std::string> &baseline,
                              const std::vector<std::string> &results)
  const override {
    // your custom comparison implementation
  }
protected:
  virtual flit::Variant run_impl(const std::vector<T> &ti) override {
    FLIT_UNUSED(ti);
    return flit::Variant();
  }
protected:
  using flit::TestBase<T>::id;
};
REGISTER_TYPE(MpiAppTest);
```

Your basic FLiT test with no implementation of `compare()` or `run_impl()`.

```c++
template<>
flit::Variant MpiAppTest<double> run_impl(const std::vector<double> &ti) {
  FLIT_UNUSED(ti);
  auto results = flit::call_mpi_main(
      mpiapp_main, "mpirun -n 4", "mpi_app", "--data ../data/star.dat");
  std::vector<std::string> vec_results;
  if (results.ret != 0) {
    throw std::logic_error("mpi_app failed in its execution");
  }
  vec_results.push_back(results.out);
  vec_results.push_back(results.err);
  return vec_results;
}
```

We only define the `double` precision test and we call `mpiapp_main` under a
child process of `mpirun` made into four processes.  The app will look like it
is called `mpi_app` (i.e., that will be the value of `argv[0]`).  Finally, the
command-line arguments (after `args[0])`are "--data", and "../data/star.dat".

The returned value from `flit::call_mpi_main()` is a `flit::ProcessResult`
struct that has standard output, standard error, and the return status.  Using
these values and possibly files created or written from `mpiapp_main()`.


[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](mpi-support.md)
