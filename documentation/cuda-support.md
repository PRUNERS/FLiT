# CUDA Support

[Prev](mpi-support.md)
|
[Table of Contents](README.md)
|
[Next](run-wrapper-and-hpc-support.md)

FLiT historically had limited CUDA support, so much so that it was eventually
stripped out for not being of enough utility and being left in disrepair.  It
may be good to have full CUDA support with reproducibility tests written with
CUDA kernels, but that is a long way off at this point.

Instead, a much more attainable goal is to build infrastructure to wrap a
compiler wuth the `nvcc` compiler to allow for tests and code bases that
utilize CUDA kernels.  This has not yet been implemented, but is on the backlog
of tasks (see [issue #164](https://github.com/PRUNERS/FLiT/issues/164)).

Unfortunately at this point, FLiT does not have CUDA support.


[Prev](mpi-support.md)
|
[Table of Contents](README.md)
|
[Next](run-wrapper-and-hpc-support.md)
