# Standard C++ Library Implementations

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](writing-test-cases.md)

Ever since C++11, it seems every compiler vendor has written their own C++
standard library.  They have moved away from `glibc`.  GCC has implemented
their own `libstdc++` library.  LLVM has implemented their own `libc++`
library.  Intel by default uses the GCC implementation from the system path.

Fortunately, there are ways to specify to each compiler which standard library
to use.

## How to Specify Standard Library

If you're only using one compiler or one type of compiler, you can simply
specify the compiler and linker flags in `custom.mk` with the `CXXFLAGS` and
`LDFLAGS` variables.

However, if you have flags that you only want to apply to a single compiler
instead of all of them, then you can make use of `flit-config.toml`.
Specifically, there are two settings under the `[[compiler]]` section used for
each specified compiler.

```toml
[[compiler]]
...
fixed_compile_flags = ''
fixed_link_flags = ''
...
```

The `fixed_compile_flags` is just a string of flags to use for every object
file compilation and for link-time too.  You can put special includes, defines,
warnings, or whatever else is specific to this compiler.

The `fixed_link_flags` are used only for link-time.  Here, you specify
libraries to link in and any other linker options specific to this compiler.

The next few sections will cover examples of values for these two settings for
using different standard libraries.

**Note:** This is not an exhaustive guide.  More like some examples of how to
get around the problem of the standard C++ library.  For sure you may find more
or better information online.

## GCC Compiler

For the GCC compiler, we can specify to use the `libc++` standard library
implementation from LLVM.  Even though it can be done, it is unfortunate that
GCC does not make it easy.  Instead, we need to undo the flags GCC
automatically puts in and manually specify the ones we need.

To undo compiler flags, we specify `-nostdinc++`.  To undo linker flags, we
specify `-nodefaultlibs`.  Then, we need to manually add back in the necessary
include and linker flags.

This amounts to the following:

```toml
[[compiler]]
...
fixed_compile_flags = '-nostdinc++ -I<libcxx-install-prefix>/include/c++/v1'
fixed_link_flags = '-nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc'
...
```

where `<libcxx-install-prefix>` is wherever the header files are installed.


## Clang Compiler

For the Clang compiler, we can specify to use the `libstdc++` standard library
implementation from GCC.  They make this pretty easy to do.  In fact, it is
usually the default behavior on Linux (but not for OS X since Mavericks).  This
is to help with compatibility since most libraries are compiled using GCC on
Linux.

To specify a specific version of the GCC standard library to use, you specify
the `--gcc-toolchain` option.  The value given to this is the install prefix
when installing GCC.  For most packages that are installed with package
managers, this is simply `/usr`.  The way to get this value is to give one
directory up from the `g++` binary you want to use.  So for `/usr/bin/g++`, you
would use `--gcc-toolchain=/usr`.  For `/opt/compilers/gcc-9.1.0/bin/g++`, you
would use `--gcc-toolchain=/opt/compilers/gcc-9.1.0`.  No additional flags are
required.

So this would look like the following:

```toml
[[compiler]]
...
fixed_compile_flags = '--gcc-toolchain=/opt/compilers/gcc-9.1.0'
...
```

If you need to have pre C++11 compatibility, then you also need to add a macro
definition to your compile flags, specifically `'-D_GLIBCXX_USE_CXX11_ABI=0'`.
This is the case whenever using `libstdc++` from GCC.


## Intel Compiler

The Intel compiler by default uses `libstdc++` from the GCC found in the system
path.  It doesn't seem to have consistent support for using `libc++` from LLVM.
It seems it only supports the `-stdlib` flag on Mac OS X.  So, if using the
Intel compiler on Linux, it seems you are stuck with `libstdc++`.

That being said, there is a way to specify which version of GCC you wish to use
with `--gcc-name` and `--gxx-name`.  I assume the former is for the C standard
library, and the latter is for the C++ standard library.  The argument to these
are the compiler binaries themselves.  So, again for our example of our GCC
compiler at `/opt/compilers/gcc-9.1.0/bin/g++`, we would specify the following
in our `flit-config.toml` file under the intel compiler entry:

```toml
[[compiler]]
...
fixed_compile_flags = '--gxx-name=/opt/compilers/gcc-9.1.0/bin/g++'
...
```

The Intel compiler will automatically insert the correct include and linker
flags.  I am unsure if the `--gcc-names` argument is really required.  I
believe only `--gxx-name` is required to do what you want when compiling C++
code.  I imagine `--gcc-name` is for when you are compiling C code with `icc`.

[Prev](available-compiler-flags.md)
|
[Table of Contents](README.md)
|
[Next](writing-test-cases.md)
