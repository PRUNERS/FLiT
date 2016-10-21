#!/bin/env bash

#SBATCH --time=1:00:00 # walltime, abbreviated by -t
#SBATCH --exclusive
#SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
###SBATCH --ntasks=
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
# additional information for allocated clusters
#SBATCH --account=kingspeak-gpu     # account - abbreviated by -A
#SBATCH --partition=kingspeak-gpu  # partition, abbreviated by -p
#SBATCH --gres=gpu:titanx:1

#
# set data and working directories
export WORKDIR=$HOME
export SCRDIR=/scratch/kingspeak/serial/u0422778
if [ ! -e "$SCRDIR" ]; then
    mkdir -p $SCRDIR
fi

cp -r $WORKDIR/remote_qfp/qfp $SCRDIR/qfp

if [ ! -e $SCDIR/software ]; then
    mkdir -p $SCRDIR/software
fi

if [ ! -e $SCDIR/software/clang/bin ]; then
    cp -r $WORKDIR/software/clang/build/bin $SCRDIR/software/clang/bin
fi

if [ ! -e $SCDIR/software/clang/lib ]; then
    cp -r $WORKDIR/software/clang/build/lib $SCRDIR/software/clang/lib
fi

export PATH=$SCRDIR/software/clang/bin:$PATH

git pull
git checkout unified_script
cd $SCRDIR/qfp/qfpc

module load gcc/4.9.2
module load intel/2016.3.210
module load cuda/7.5

#
# load appropriate modules, in this case Intel compilers, MPICH2
#module load intel mpich2
# for MPICH2 over Ethernet, set communication method to TCP
# see below for network interface selection options for different MPI distributions
#setenv MPICH_NEMESIS_NETMOD tcp
# run the program
# see below for ways to do this for different MPI distributions
#mpirun -np $SLURM_NTASKS my_mpi_program > my_program.out

export DEVCAP=-gencode=arch=compute_52,code=sm_52
if [ -z "$SLURM_NTASKS" ]; then
    PROCS=24
else
    PROCS=$SLURM_NTASKS
fi

make -j $PROCS &> slurmOut

cp -r ../results/* $HOME/remote_qfp/qfp/results
cp slurmOut $HOME/remote_qfp/qfp/results/slurmOut_$HOST



