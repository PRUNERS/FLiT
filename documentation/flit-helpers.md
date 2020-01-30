# FLiT Helpers

[Prev](writing-test-cases.md)
|
[Table of Contents](README.md)
|
[Next](mpi-support.md)

The FLiT library has a number of utility functionality that may be useful to
those writing tests.

- [CSV Reader and Writer](#csv-reader-and-writer)
- [Info Stream](#info-stream)
- [File System Utilities](#file-system-utilities)
- [Subprocess Utilities](#subprocess-utilities)
- [Miscellaneous](#miscellaneous)

## CSV Reader and Writer

FLiT provides two classes for reading and writing CSV files, `flit::CsvReader`
and `flit::CsvWriter`.  These classes are used within FLiT to create the test
result files, but are usable by users.

The details can be seen in the FLiT source tree in `src/FlitCsv.h` and
`src/FlitCsv.cpp`.  Here, we give example usage of these classes.


### `flit::CsvRow`

This is the basis of the usage with `flit::CsvReader`.  This class derives from
`std::vector<std::string>` so you can use any function defined on that
interface.

Additionally, this class provides

- `flit::CsvRow::header()` returning the `CsvRow` instance that represents the
  header row.
- `flit::CsvRow::setHeader()`: sets the header row
- `flit::CsvRow::operator[](std::string)`: allowing you to index based on the
  column header name.


### `flit::CsvReader`

Reads in from a stream and returns instances of `CsvRow` (which derive from
`std::vector<std::string>`).  Here is an example of its usage:

```c++
std::istringstream input(
  "#,Cost,Description\n"
  "1,-3,Candy bar\n"
  "2,-5,Garage parking\n"
  "3,315,Got paid!\n"
  );
Flit::CsvReader reader(input);
int balance = 0;
for (flit::CsvRow row; reader >> row;) {
  balance += std::stoi(row["Cost"]);
  // could also do
  // balance += std::stoi(row[1]);
}
assert(balance == 307);
```

There are other functions not demonstrated in this example:

- `flit::CsvReader::header()`: returns a `CsvRow` containing the header row
- `flit::CsvReader::stream()`: returns the stream that was passed in and is
  being used
- casting `flit::CsvReader` to a `bool` evaluates to `false` if there are no
  more rows to parse and evaluates to `true` otherwise.


### `flit::CsvWriter`

Writes to a stream in CSV format.  Here is an example of its usage:

```c++
std::ofstream out("output.csv");
flit::CsvWriter writer(out);

std::vector<std::vector<double>> data {
  {1.0, 2.0, 3.3},
  {2.0, 2.0, 5.2},
  {3.0, 2.5, 7.7},
};

writer << "x" << "y" << "z";
writer.new_row();          // start a new row
writer << data[0][0]
       << data[0][1]
       << data[0][2];
writer.new_row();          // start a new row
writer.write_row(data[1]); // write the whole row all at once
writer.write_row(data[2]); // write the whole row all at once
```

Supported types with `operator<<()` are

- `int`
- `long`
- `long long`
- `unsigned int`
- `unsigned long`
- `unsigned long long`
- `unsigned __int128`
- `float`
- `double`
- `long double`
- `std::string`

To implement your own types, you will need to convert to one of these types.
The `std::string` type will probably be the most common route.

## Info Stream

FLiT provides a way to output debugging information from tests that will only be printed when the `--verbose` flag is given to the compiled FLiT tests.  This is through a singleton object called `flit::info_stream`.  This object is an instance of the `flit::InfoStream` class that derives from `std::ostream`, so any object that can be printed to `std::ostream` can be printed with `flit::info_stream`.  This stream is thread-safe.

Here is an example usage from within a test:

```c++
virtual flit::Variant run_impl(const std::vector<T>& ti) override {
  auto &info = flit::info_stream;
  info << id << ": beginning test\n";

  // test implementation ...

  info << id << ": ending test\n";
  info << id << ": returned value = " << retval << "\n";
  return retval;
}
```

This class has a few more methods of interest:

- `flit::InfoStream::show()`: turns on the debugging messages.  This is the
  same as calling the tests with the `--verbose` flag.  Note: each thread has
  their own local copy of `flit::info_stream`, so this would only affect the
  current thread.
- `flit::InfoStream::hide()`: the opposite of `show()`
- `flit::InfoStream::flushout()`: flushes the stream


## File System Utilities

It is common to work with files within tests, so many of these functionalities
are made easier if desired.  These are defined in `src/fsutil.h` and
`src/fsutil.cpp`.


### `flit::ifopen()` and `flit::ofopen()`

Open files for in and out with error checking.  By default, if you open a file
with `std::ifstream` and the file does not exist, or you open a file with
`std::ofstream` with permission errors, then the default behavior is to
silently ignore those errors.  Since that is not too useful to a programmer,
these utility functions open the files with error checking turned on.

```c++
std::ifstream in;
flit::ifopen(in, "in-file.txt"); // will throw a std::ios::failure if it fails

std::ofstream out;
flit::ofopen(out, "out-file.txt"); // will throw if it fails
```

Also, the `flit::ofopen()` function sets the precision of the `std::ofstream`
to a very high number to ensure outputting does not cause floating-point
truncation.


### `flit::join()`

Joins two parts of a path together.  Currently, the only supported separator is
the Unix one, "/", but if Windows is supported in the future, this function
would join path elements properly.

Note, you can pass in as many arguments to this function as you want.

```c++
auto full_path = flit::join("/home", "user", "git", "FLiT", "src", "fsutil.h");
```


### `flit::readfile()`

Reads the entire content of the given filename into a `std::string`.

```c++
std::string contents = flit::readfile(flit::join("data", "setup.txt"));
```


### `flit::listdir()`

Lists the contents of a directory into a `std::vector<std::string>`.

```c++
auto listing = flit::listdir(".");
```


### `flit::printdir()`

Prints the contents of a directory to the console.

```c++
flit::printdir("."); // prints all files and folders in the current directory
```


### `flit::touch()`

Creates an empty file if it doesn't exist, or update the modification time if
it does exist.  Similar to the `touch` command-line utility.

```c++
flit::touch("empty.txt"); // create a new empty file
```


### `flit::mkdir()`

Make a directory (optionally with a unix permission mode).  Throws an exception on failure (`std::ios_base::failure`).

```c++
flit::mkdir(flit::join("output", "directory"), 0777);
```

The above will attempt to make the directory "output/directory" with read,
write, and execute permissions for all.


### `flit::rmdir()`

Remove an empty directory.  Throws a `std::ios_base::failure` exception on
failure (which will happen if the directory is not empty, you do not have
permission to delete it, or it doesn't exist).

```c++
flit::rmdir(flit::join("output", "directory"));
```


### `flit::rec_rmdir()`

Recursively delete everything under a particular directory.  This is the same
as `rm -rf <directory>` in the console.  Throws a `std::ios_base::failure`
exception upon failure (mostly for permission issues).

```c++
using flit::join;
flit::mkdir("output");
flit::mkdir(join("output", "directory"));
flit::touch(join("output", "directory", "file1.txt"));
flit::touch(join("output", "file2.txt"));
flit::rec_rmdir("output"); // delete it all
```


### `flit::curdir()`

Simply returns the absolute path to the current working directory.

```c++
auto current_path = flit::curdir();
```


### `flit::chdir()`

Change the to given directory (i.e. `cd` on the command-line)

```c++
auto prev_path = flit::curdir();
flit::mkdir("output");
flit::chdir("output");
// do some work in output
flit::chdir(prev_path);
```


### `flit::which()`

Finds the given filename from a list of paths (given as a colon-separated
list).  If the path is not given, then the `PATH` environment variable is used.

```c++
auto system_echo = flit::which("echo");
auto my_echo = flit::which("echo",
                           "/home/user/bin:/home/user/sys/bin:/opt/bin");
```

The first one uses the system `PATH` variable for lookup (returning the same
thing as `which echo` from the command-line).  The second is given a
user-specified path string with colon separated search paths.

If no such file is found, then `std::ios_base::failure` is thrown.


### `flit::realpath()`

Returns the cononical absolute path after resolving symbolic links of the given
path.  The given path need not exist.  For those elements that do exist and are
symbolic links, they are resolved.

```c++
auto absolute = flit::realpath(os.path.join(".", "output", "..", "output",
                                            "test.txt"));
```

This will resolve as expected.


### `flit::dirname()`

Returns the parent directory path of the given path.

```c++
auto parent = flit::dirname("a/b/c");
```

The above will return `"a/b"` as the parent.


### `flit::basename()`

Returns the last component of a path, which may be a file or a directory.

```c++
auto filename = flit::basename("dir/subdir/file.txt");
```

The above will result in `"file.txt"`.


### Class `TempDir`

An RAII-style class that creates a temporary directory.  When the class goes
out of scope (i.e., is deleted), then the directory, along with all of its
contents, are deleted.

```c++
{
  flit::TempDir tmpdir;
  flit::mkdir(flit::join(tmpdir.name(), "data"));
  std::ofstream out;
  flit::ofopen(out, flit::join(tmpdir.name(), "data", "output.txt"));
  // output to out
  // do some work ...
}
// when we go out of scope, tmpdir and its contents will be deleted
```


### Class `TempFile`

An RAII-style class that creates a temporary file.  When the class goes out of
scope (i.e., is deleted), then teh file is deleted.  Optionally, you can give a
directory where to create the temporary file.

```c++
{
  flit::TempDir tmpdir;
  flit::TempFile tmpfile(tmpdir.name());
  tmpfile.out << "hello there";
  std::cout << "Created " << tmpfile.name << std::endl;
}
```


### Class `PushDir`

An RAII-style class that calls `flit::chdir()` in the constructor to the given
directory, and calls `flit::chdir()` back to the directory it was in before.
You can also query what the previous directory was.

```c++
{
  flit::TempDir temporaryDirectory;
  flit::PushDir pusher(temporaryDirectory.name());
  // now in the temporary directory
  // do some work ...
}
// first pusher goes out of scope and we cd out of temporaryDirectory
// then temporaryDirectory goes out of scope and the directory is deleted
```


### Class `FileCloser`

An RAII-style class for handling the closing of a `FILE*` pointer (created
using the `std::fopen()`-family of functions).  When this class goes out of scope,
it will close the `FILE*` pointer, preventing resource leaks in the case of
exceptions or programmer negligence.

```c++
{
  flit::FileCloser handler(std::fopen("hello.txt", "w"));
  fprintf(handler.file, "Value: %02f\n", 3.14159);
}
```

Also gives access to the integer `fd` file descriptor.  This can be useful as
shown in the next section with the `FdReplace` class.


### Class `FdReplace`

An RAII-style class for replacing the file descriptor of one `FILE*` handle
with the file descriptor of another `FILE*` handle.  When this object goes out
of scope and is deleted, then the original file descriptor is restored.

An example usage of this is to capture output that would go to stdout.  If you
use `std::cout`, then there are ways to redirect that output to a stream buffer
for another stream, such as a `std::stringstream` and capture it that way.  But
what about code that calls `printf()` and other functions that act on the
`stdout` or `stderr` global pointers?  Here is a trick to grab that output.
And it seems most implementations of `std::cout` end up feeding into the
`stdout` pointer, so capturing it here gets all of it.

```c++
{
  flit::FileCloser stdout_handler(std::fopen("stdout-captured.txt", "w"));
  flit::FdReplace stdout_capturer(stdout, stdout_handler.file);
  printf("hello world!\n"); // will print instead to "stdout-captured.txt"
  std::cout << "hello world!\n"; // will also be captured
}
// stdout is now restored
```


### Class `StreamBufReplace`

An RAII-style class for replacing stream buffers.  Suppose you do not want your
function to output to `std::cout` during the test, but at runtime, you do.  In
your test, you can capture the output using this class, and restore it at the
end of your test.

```c++
{
  std::ostringstream capturer;
  flit::StreamBufReplace replacer(std::cout, capturer);
  std::cout << "hello world!\n"; // will be captured by capturer
  capturer.str(); // will contain "hello world!\n"
}
// std::cout is now restored
std::cout << "hello again\n"; // will be printed to the screen
```


## Subprocess Utilities

Many tests may wish to call subprocesses.  There are already utilities such as
the `system()` call.  These are a little hard to use, especially if you want to
capture the standard output of the child process.  Fortunately, FLiT has some
useful helper functions for you.

These utilities are defined in `src/subprocess.h` and `src/subprocess.cpp`.


### `flit::call_with_output()`

Calls a command as though from the command-line and returns a
`flit::ProcResult` struct containing the returncode, stdout, and stderr of the
child process.

```c++
auto result = flit::call_with_output("curl localhost:8000/data.html");

if (result.ret != 0) {
  throw std::logic_failure("curl command failed");
}

flit::info_stream << "curl stdout:\n" << result.out << "\n\n"
                  << "curl stderr:\n" << result.err << "\n\n";
```


### Calling and Registering a Main Function

This topic is also covered in
[Writing Test Cases](writing-test-cases.md#wrapping-around-a-main-function).

It is often the case that you want to test your application as a whole.  That
means calling and testing your `main()` function directly.  Here is an example
to illustrate how to use the functions declared in `src/subprocess.h` to run a
`main()` function.

```c++
#include <flit/flit.h>

#define main my_main
#include "main.cc"
#undef main
FLIT_REGISTER_MAIN(my_main)

template <typename T>
class MainTest : public flit::TestBase<T> {
   ...
};
REGISTER_TYPE(MainTest);

template<>
flit::Variant MainTest<double> run_impl(const std::vector<double> &ti) {
  FLIT_UNUSED(ti);
  auto results = flit::call_main(my_main, "myapp", "--input data/file.txt");
  std::vector<std::string> vec_results;
  if (results.ret != 0) {
    throw std::logic_error("my_main() failed in its execution");
  }
  vec_results.emplace_back(std::move(results.out));
  vec_results.emplace_back(std::move(results.err));
  return vec_results;
}
```

More detail is given in
[Writing Test Cases](writing-test-cases.md#wrapping-around-a-main-function).
But, here we will highlight a few points.  First, the file containing the user-defined `main()` function is in `main.cc`, which is included into the test file after the `#define main my_main`.  That trick renames the name `main` to `my_main`.  It no longer name clashes, our app still has only one `main()`, and we can call `my_main()` as a normal function now.  After including that source file, we call `FLIT_REGISTER_MAIN()` on the newly named `my_main()` function for later.

In our test, we call `flit::call_main()`.  In it we give three parameters:

1. the function pointer to the `main`-like function.  This function pointer
   needs to be registered with `FLIT_REGISTER_MAIN()`.  This is so that FLiT
   can call this function in a child process.
2. the value of `argv[0]` when `my_main()` is called (in a child process).
   Some programs change their behavior when the name of the executable changes
   (for example, `bash` behaves different if the name of the executable is
   `sh`).
3. the rest of the command-line parameters as would be given on the bash
   prompt.

Afterwards, the standard output and standard error are placed into a vector and
returned.  It will be the job of the user's `compare()` function that will make
sense of the standard output and standard error when doing the comparison
between two runs.

Similarly, there is a function called `call_mpi_main()` that has a very similar
interface to `call_main()`.  The only difference is that an extra argument is
inserted as the second argument.  This extra argument is the command (with
arguments) for the `mpirun` executable.  Not only can you choose which `mpirun`
you want to use (or if you want to use `srun` or abuse this function to call a
different wrapper than has nothing to do with MPI), but you can specify the
arguments to `mpirun` such as the number of processes to
create.


## Miscellaneous

There are many other helper functions that may be useful to test writers.
Those are documented here, along with where they are implemented.

### `flit::split()` (in `src/flitHelpers.h`)

Split a string by a delimiter.  This is similar to python's `split()` function.

```c++
auto split_by_space = flit::split("my name is Mike", ' ');
for (auto &item : split_by_space) {
  std::cout << "'" << item << "'" << std::endl;
}
```

This will print

```
'my'
'name'
'is'
'Mike'
```

There is an optional parameter of `maxsplit` which specifies how many times to
split maximum.

```c++
auto split_only_twice = flit::split("flit::CsvReader::operator<<()", ':', 2);
for (auto &item : split_only_twice) {
  std::cout << "'" << item << "'" << std::endl;
}
```

This will print

```
'flit'
''
'CsvReader::operator<<()'
```

### `flit::rtrim()`, `flit::ltrim()`, and `flit::trim()` (in `src/flitHelpers.h`)

All three of these functions return new strings so that the original is not
modified.  This is to prevent unintended consequences in code.

- `flit::rtrim()`: trims whitespace from the right-hand side of a string
- `flit::ltrim()`: trims whitespace from the left-hand side of a string
- `flit::trim()`: trims whitespace from both the left and right


### `flit::rstrip()`, `flit::lstrip()`, and `flit::strip()` (in `src/flitHelpers.h`)

These three are similar to the `flit::trim()` functions.  These, however, can
strip off any amount of an arbitrary string from the right, left, or both
sides, respectively.

- `flit::rstrip()`: strip any amount of a given string from the right and return
- `flit::lstrip()`: strip any amount of a given string from the left and return
- `flit::strip()`: strip from both sides

```c++
auto stripped = flit::rstrip("-*--*- hello -*--*-", "-*-");
```

This will result in `"-*--*- hello "`, since only the right side was stripped.


### `flit::l2norm()` (in `src/flitHelpers.h`)

Computes the L2 norm between two `std::vector<T>` instances.  The type of `T`
must implement minus and multiply between themselves, and they must implement
plus with a `long double` accumulator.  This is intended for floating-point
types, but users can use custom types if they choose.

[Prev](writing-test-cases.md)
|
[Table of Contents](README.md)
|
[Next](mpi-support.md)

