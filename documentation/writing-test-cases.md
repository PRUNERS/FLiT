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

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](database-structure.md)
