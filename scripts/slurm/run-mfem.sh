#!/bin/bash -l
#SBATCH --time 48:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 28
#SBATCH --cpus-per-task 1
#SBATCH --account soc-kp
#SBATCH --partition soc-kp

set -e
set -u
set -x

ml singularity
ml

scratch=/scratch/kingspeak/serial/u0415196

# Setup and build all of the tests
singularity exec ~/singularity/archflit-mfem.simg \
  bash -c \
    "cp -r /opt/mfem /tmp && \
     flit update -C /tmp/mfem/flit_tests/ && \
     make -j56 ground-truth.csv dev runbuild -C /tmp/mfem/flit_tests"

# Since the tests are run using mpirun, we can't run tests naively,
# otherwise we will only be able to use two physical cores at
# a time, and all jobs will be using those two physical cores.
# So instead, we use srun for each one in parallel

# Run the tests
# The --mem-per-cpu=0 is special and indicates to have steps within the job
# allocation should share the total memory allocation.  Each one will have
# access to the full memory allocation at the same time.
cd /tmp/mfem/flit_tests

find results -type f -executable | \
  parallel -j20 --ungroup \
    srun -n 1 --mem-per-cpu=0 --exclusive \
      singularity exec ~/singularity/archflit-mfem.simg \
        bash -c \
          "\"make -j1 {}-out-comparison.csv && \
             cp {}-out {}-out-comparison.csv ~/mfem-results/\""

