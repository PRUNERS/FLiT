#!/bin/bash -l
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 28
#SBATCH --cpus-per-task 1
#SBATCH --account owner-guest
#SBATCH --partition kingspeak-guest

### %#SBATCH --account ganesh-kp
### %#SBATCH --partition soc-kp

ml singularity
echo "Step 1: show what is installed"
ml
singularity exec ~/singularity/archflit-mfem.simg \
  bash -c \
    "g++ --version && \
     clang++ --version && \
     icpc --version"

echo -e "\n\n\n\n\n\n\n\n"
echo "Step 2: compile tests"

# Setup and build gtrun and devrun
singularity exec ~/singularity/archflit-mfem.simg \
  bash -c \
    "cp -r /opt/mfem /tmp && \
     flit update -C /tmp/mfem/flit_tests/ && \
     make -j56 gt dev -C /tmp/mfem/flit_tests/"

echo -e "\n\n\n\n\n\n\n\n"
echo "Step 3: run tests"

cd /tmp/mfem/flit_tests

# Worked:
#parallel -j10 --ungroup \
#  "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Didn't work:
#parallel -j10 --ungroup \
#  srun -n 1 \
#    singularity exec ~/singularity/archflit-mfem.simg \
#      mpirun -n 1 ./gtrun -o gtrun-{}.csv \
#  ::: {01..50}

# Didn't work:
# it ran 10 of them, but all sharing one core
#parallel -j10 --ungroup \
#  srun -n 1 --mem-per-cpu=0 \
#    singularity exec ~/singularity/archflit-mfem.simg \
#      mpirun -n 1 ./gtrun -o gtrun-{}.csv \
#  ::: {01..50}

# Worked:
#srun="srun -n 1 --mem-per-cpu=0 --exclusive"
#sing="singularity exec $HOME/singularity/archflit-mfem.simg"
#mpi="mpirun -n 1"
#$srun $sing $mpi ./gtrun -o gtrun-01.csv &
#$srun $sing $mpi ./gtrun -o gtrun-02.csv &
#$srun $sing $mpi ./gtrun -o gtrun-03.csv &
#$srun $sing $mpi ./gtrun -o gtrun-04.csv &
#$srun $sing $mpi ./gtrun -o gtrun-05.csv &
#wait

# WORKED!
#parallel -j10 --ungroup \
#  srun -n 1 --mem-per-cpu=0 --exclusive \
#    singularity exec ~/singularity/archflit-mfem.simg \
#      mpirun -n 1 ./gtrun -o gtrun-{}.csv \
#  ::: {01..50}

# Didn't work:
# Failed with errors:
#   srun: error: Invalid user for SlurmUser slurm, ignored
#   srun: fatal: Unable to process configuration file
#parallel -j10 --ungroup \
#  singularity exec -B /etc/sysconfig -B /uufs/kingspeak.peaks/sys ~/singularity/archflit-mfem.simg \
#    /uufs/kingspeak.peaks/sys/pkg/slurm/std/bin/srun -n 1 --mem-per-cpu=0 --exclusive ./gtrun -o gtrun-{}.csv \
#  ::: {01..50}

# Didn't work
#parallel -j10 --ungroup \
#  srun -n 1 \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Didn't work
#parallel -j10 --ungroup \
#  srun -n 1 --cpu-bind=rank \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Didn't work
#parallel -j10 --ungroup \
#  srun -n 1 --cpu-bind=verbose \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# WORKED!!!
#parallel -j10 --ungroup \
#  srun -n 1 --mem-per-cpu=4G \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# WORKED!!!
#parallel -j10 --ungroup \
#  srun -n 1 --mem-per-cpu=0 \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Didn't work
#export SLURM_CPU_BIND=none
#parallel -j10 --ungroup \
#  srun -n 1 \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Didn't work
#parallel -j10 --ungroup \
#  srun -n 1 --exclusive \
#    "md5sum /dev/urandom > /tmp/random-{}.md5" \
#  ::: {01..50}

# Failed to run srun within singularity container
#cp -r /uufs/kingspeak.peaks/sys/pkg/slurm/std/ /tmp/slurm
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/slurm/lib:/tmp/slurm/lib/slurm
#
#parallel -j10 --ungroup \
#  singularity exec ~/singularity/archflit-mfem.simg \
#    /tmp/slurm/bin/srun -n 1 ./gtrun -o gtrun-{}.csv \
#  ::: {01..50}
