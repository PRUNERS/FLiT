#!/usr/bin/env bash

#SBATCH --time=1:00:00 # walltime, abbreviated by -t
#SBATCH --exclusive
#SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
#SBATCH --ntasks=8
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
# additional information for allocated clusters
#SBATCH --account=kingspeak-gpu     # account - abbreviated by -A
#SBATCH --partition=kingspeak-gpu  # partition, abbreviated by -p
#SBATCH --gres=gpu:titanx:1

#
# set data and working directories
setenv WORKDIR $HOME/remote_qfp/qfp
setenv SCRDIR /scratch/kingspeak/serial/u0422778/qfp
mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
git pull
git checkout rel_lt_cuda
cd $SCRDIR/qfpc

module load gcc/5.4.0
module load intel/2016.0.3.210
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

make -j $SLURM_NTASKS &> slurmOut

cp -r ../results/* $HOME/remote_qfp/qfp/results
cp slurmOut $HOME/remote_qfp/qfp/results/slurmOut_$HOST



