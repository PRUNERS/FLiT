# Writing Test Cases

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](database-structure.md)

When you initialize a test directory using `flit init`, a test file called
`Empty.cpp` will be placed in the `tests` directory.  This test is as the name
implies - empty.  It basically shows you how to create a new test and is
intended to be used as a template for creating tests.

To add a test, simply copy `Empty.cpp` and place it in the `tests` directory
with an ending of `cpp`.  Rename the class inside, and you're good to go.

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
virtual flit::Variant run_impl(const flit::TestInput<T>& ti) override {
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
flit::Variant MyTestClass<double>::run_impl(const flit::TestInput<double>& ti) {
   // test logic here ...
   return something_meaningful;
}
```

Using this approach, you can have this test only implemented for `double`
precision, and have the other precisions disabled.  That way, you will not have
misleading information of your test case using `float` or `long double`
precision taking up space in your results database.

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](database-structure.md)
