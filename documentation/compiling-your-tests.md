# Compiling Your Tests

[Prev](run-wrapper-and-hpc-support.md)
|
[Table of Contents](README.md)
|
[Next](test-executable.md)


Probably the hardest aspect of integrating any testing framework into your
project is getting the tests to compile against the rest of your code base.
This is especially true for FLiT.  Since much of the FLiT framework is built
into the auto-generated `Makefile`, it is important for you to give this
auto-generated `Makefile` enough information to compile your source code.

This has been done and tested with two separate open-source projects, and here
I give the details of these two examples, hopefully to give you insight into
how to accomplish this for your own project.

## Explaining the `custom.mk` File

The `custom.mk` file is included by the generated `Makefile`.  It gives you a
portal to modify some of the `Makefile` behaviors.  It is a bad idea to modify
the generated `Makefile` because it may get regenerated and overwrite your
changes.  Instead put changes in this `custom.mk` file.

The `custom.mk` file is already populated with variables that you may want to
use.  Simply add to these variables to compile your application.

- `SOURCE`: This variable contains source code to compile other than the tests
  that are in the `tests` directory.
- `CC_REQUIRED`: Compiler flags to use across the board, for the dev build,
  ground-truth build, and all other builds.  Here is where you set macro
  variables, include paths, and perhaps warning flags.
- `LD_REQUIRED`: Linker flags to use across the board, for the dev build,
  ground-truth build, and all other builds.  Here is where you include library
  paths (with `-L`), link in libraries (with `-l`), and maybe even some rpath
  magic (with `-Wl,-rpath=<path>`).
- `DEV_CFLAGS`: Extra compiler flags specifically for the dev build.  This will
  be added to the `CC_REQUIRED` when compiling the dev target.
- `DEV_LDFLAGS`: Extra linker flags specifically for the dev build.  This will
  be added to the `LD_REQUIRED` when compiling the dev target.
- `RUN_WRAPPER`: a program to wrap all invocations of test executable runs.
  This gives the user the opportunity to control test executions if desired.


## MFEM Example

[MFEM](http://mfem.org) is a modular parallel C++ library for finite element
methods.  The source code can be found on
[Github](https://github.com/mfem/mfem.git).  Here, I will go through my fork of
the MFEM code base where I implemented FLiT tests.  This forked project can be
pulled with

```bash
git clone --branch flit-examples https://github.com/mikebentley15/mfem.git
```

The tests are then located in `mfem/examples/flit-tests`.  The tests are
actually the examples found in `mfem/examples`, and taking their `main()`
function and modifying it to become a FLiT test case.  Not all of the examples
could be converted in this way.  Examples 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14,
15, 16, and 17 were able to be converted while examples 11, 12, and 13 were
not.

This project was easier to compile than I though it would be.  A `custom.mk`
file that would work to compile this would be:

```make
SOURCE         += $(wildcard ../../fem/*.cpp)
SOURCE         += $(wildcard ../../general/*.cpp)
SOURCE         += $(wildcard ../../linalg/*.cpp)
SOURCE         += $(wildcard ../../mesh/*.cpp)

CC_REQUIRED    += -I../..
```

It turns out that not all of those source files were required for the given
examples, meaning we will be compiling more than is necessary.  Using trial and
error, I determined exactly which files need to be included to successfully
compile and no less:

```make
SOURCE         += ../../fem/bilinearform.cpp
SOURCE         += ../../fem/bilininteg.cpp
SOURCE         += ../../fem/coefficient.cpp
SOURCE         += ../../fem/datacollection.cpp
SOURCE         += ../../fem/eltrans.cpp
SOURCE         += ../../fem/estimators.cpp
SOURCE         += ../../fem/fe.cpp
SOURCE         += ../../fem/fe_coll.cpp
SOURCE         += ../../fem/fespace.cpp
SOURCE         += ../../fem/geom.cpp
SOURCE         += ../../fem/gridfunc.cpp
SOURCE         += ../../fem/hybridization.cpp
SOURCE         += ../../fem/intrules.cpp
SOURCE         += ../../fem/linearform.cpp
SOURCE         += ../../fem/lininteg.cpp
SOURCE         += ../../fem/nonlinearform.cpp
SOURCE         += ../../fem/nonlininteg.cpp
SOURCE         += ../../fem/staticcond.cpp        # 17/24 of the fem files
SOURCE         += ../../general/array.cpp
SOURCE         += ../../general/error.cpp
SOURCE         += ../../general/gzstream.cpp
SOURCE         += ../../general/socketstream.cpp
SOURCE         += ../../general/stable3d.cpp
SOURCE         += ../../general/table.cpp
SOURCE         += ../../general/tic_toc.cpp       #  7/12 of the general files
SOURCE         += ../../linalg/blockmatrix.cpp
SOURCE         += ../../linalg/blockoperator.cpp
SOURCE         += ../../linalg/blockvector.cpp
SOURCE         += ../../linalg/densemat.cpp
SOURCE         += ../../linalg/matrix.cpp
SOURCE         += ../../linalg/ode.cpp
SOURCE         += ../../linalg/operator.cpp
SOURCE         += ../../linalg/solvers.cpp
SOURCE         += ../../linalg/sparsemat.cpp
SOURCE         += ../../linalg/sparsesmoothers.cpp
SOURCE         += ../../linalg/vector.cpp         # 11/17 of the linalg files
SOURCE         += ../../mesh/element.cpp
SOURCE         += ../../mesh/hexahedron.cpp
SOURCE         += ../../mesh/mesh.cpp
SOURCE         += ../../mesh/mesh_operators.cpp
SOURCE         += ../../mesh/mesh_readers.cpp
SOURCE         += ../../mesh/ncmesh.cpp
SOURCE         += ../../mesh/nurbs.cpp
SOURCE         += ../../mesh/point.cpp
SOURCE         += ../../mesh/quadrilateral.cpp
SOURCE         += ../../mesh/segment.cpp
SOURCE         += ../../mesh/tetrahedron.cpp
SOURCE         += ../../mesh/triangle.cpp
SOURCE         += ../../mesh/vertex.cpp           # 12/16 of the mesh files

CC_REQUIRED    += -I../..
```

Reducing the files to the minimal set increases performance by approximately
40%.  That's significant.  So it may be worthwhile to do this for your project
as well.


## Laghos Example

[Laghos](https://github.com/CEED/Laghos)(LAGrangian High-Order Solver) is an
open-source miniapp that solves the time-dependent Euler equations of
compressible gas dynamics in a moving Lagrangian frame.

It was reported that experimental work in one of their branches had a
vulnerability when compiled with the IBM compiler between the optimization
levels `-O2` and `-O3`.  So it was necessary to get it to compile under FLiT in
order to diagnose the problem with [`flit
bisect`](flit-command-line.md#flit-bisect).

Again, I forked this project in order to add FLiT tests.  For this example, I
copied the `main()` function into a single FLiT test that returned the `|e|`
value after one iteration.  You can pull my fork with

```bash
git clone --branch raja-dev-flit https://github.com/mikebentley15/Laghos.git
```

I created the `custom.mk` file by studying the main `Makefile` for the Laghos
project.  I would also run the main `Makefile` with verbosity turned on (which
in this case is done with `make v=1`) so that I could see the compiler flags
sent to the compilers.  This particular `Makefile` made assumptions about
specific environment variables that would be set, but to have default values
within the `Makefile`.

```make
CUB_DIR        ?= ./cub
CUDA_DIR       ?= /usr/local/cuda
MFEM_DIR       ?= $(HOME)/home/mfem/mfem-master
RAJA_DIR       ?= $(HOME)/usr/local/raja/last
MPI_HOME       ?= $(HOME)/usr/local/openmpi/3.0.0
```

Next was to gather all relevant source files to be compiled, which in this case
was all of them except for the file with the original `main()` function, which
is `laghos.cpp`.  FLiT has its own `main()` and the functionality of the Laghos
`main()` has been put into the test called `LaghosTest`.

Again, this list was gathered from the Laghos `Makefile`.

```make
RAJA           := ../raja
KERNELS        := $(RAJA)/kernels

KERNEL_FILES   :=
KERNEL_FILES   += $(wildcard $(KERNELS)/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/blas/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/force/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/geom/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/maps/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/mass/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/quad/*.cpp)
KERNEL_FILES   += $(wildcard $(KERNELS)/share/*.cpp)

RAJA_FILES     :=
RAJA_FILES     += $(wildcard $(RAJA)/config/*.cpp)
RAJA_FILES     += $(wildcard $(RAJA)/fem/*.cpp)
RAJA_FILES     += $(wildcard $(RAJA)/general/*.cpp)
RAJA_FILES     += $(wildcard $(RAJA)/linalg/*.cpp)
RAJA_FILES     += $(wildcard $(RAJA)/tests/*.cpp)

SOURCE         += $(wildcard ../*.cpp)
SOURCE         += $(KERNEL_FILES)
SOURCE         += $(RAJA_FILES)
SOURCE         := $(filter-out ../laghos.cpp,$(SOURCE))
```

The next step is to gather required compiler flags.  This again was done by
executing the Laghos `Makefile` with verbosity turned on.  The flags that were
extracted are below:

```make
CC_REQUIRED    += -D__LAMBDA__
CC_REQUIRED    += -D__TEMPLATES__
CC_REQUIRED    += -I$(CUDA_DIR)/samples/common/inc
CC_REQUIRED    += -I$(MFEM_DIR)
CC_REQUIRED    += -I$(MFEM_DIR)/../hypre-2.10.0b/src/hypre/include
CC_REQUIRED    += -I$(MPI_HOME)/include
# Note: The local cub directory needs to be included before raja because some
#       files shadow the same header files found in raja.
CC_REQUIRED    += -I../cub
CC_REQUIRED    += -I..
CC_REQUIRED    += -I$(RAJA_DIR)/include
CC_REQUIRED    += -fopenmp
CC_REQUIRED    += -m64
```

As you can see from the note above is that we have some shadowing of header
files with this compilation.  I do not recommend creating projects that have
shadowing, but if you do, the order of include directories matters.

Next, we need to specify the linker flags.  This again was gathered by looking
at the compilation flags used by the Laghos `Makefile` when running it.

```make
LD_REQUIRED    += -L$(MFEM_DIR) -lmfem
LD_REQUIRED    += -L$(MFEM_DIR)/../hypre-2.10.0b/src/hypre/lib -lHYPRE
LD_REQUIRED    += -L$(MFEM_DIR)/../metis-4.0 -lmetis
LD_REQUIRED    += -lrt
LD_REQUIRED    += $(RAJA_DIR)/lib/libRAJA.a
LD_REQUIRED    += -Wl,-rpath -Wl,$(CUDA_DIR)/lib64
LD_REQUIRED    += -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcudadevrt -lnvToolsExt
LD_REQUIRED    += -ldl
```

Again, the order here may matter.  I didn't test that, but rather just used the
same order as the Laghos `Makefile`.

And with that, the FLiT test was able to compile against the Laghos source
code.  Note, there is setup necessary to actually run this example,
specifically all of the dependencies of Laghos for this experimental
development branch.  Since I believe that may be too far beyond the scope of
this documentation, it is omitted.


[Prev](run-wrapper-and-hpc-support.md)
|
[Table of Contents](README.md)
|
[Next](test-executable.md)

