# Intel Linker

When linking executables using the Intel compiler, by default, the following
static libraries are used instead of system dynamic libraries:

- `libdecimal.a`
- `libimf.a`
- `libipgo.a`
- `libirc_s.a`
- `libirc.a`
- `libirng.a`
- `libsvml.a`

If any of these libraries override functionality in shared libraries that are
installed on the system, there is a possibility that it can cause
reproducibility problems.  This static linking is performed by default, even if
`-O0` is used.

For example, `libimf.a` redefines many of the math functions found in
`libm.so`.

These libraries are not the same as the Intel MKL.  Yet inside of `libimf.a`,
we have definitions of functions such as `sin()`, `cos()`, and `exp()`.  When
these libraries cause reproducibility issues, it is not clear which symbols are
to blame, and I do not think there is a safe way to isolate these symbols.


## Bisect Identify Problem Library

I cannot guarantee that I can identify which symbol within the library is
causing problems, but at the very least, I can identify which of the static
libraries are to blame by doing the bisect approach.  It may be good as a first
approach to simply identify **if** extra static libraries are used and identify
**if** the static libraries cause variability.  Then, if that is true, then we
can try to identify **which** static library is to blame.

It turns out that I cannot isolate which of the static libraries are to blame
because they depend on each other.  Furthermore, if an object file is compiled
with the intel compiler, it has a very large likelihood of requiring at least
`libirc.a`, if not most of the other libraries.  I have not yet found a way to
separate out compilation with the linking of these static libraries.  We cannot
compile with intel and then _not_ link with these static libraries, it will
fail to compile.
