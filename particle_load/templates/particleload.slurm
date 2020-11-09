#!/bin/bash

#SBATCH --ntasks=N_TASKS
#SBATCH --exclusive
#SBATCH -J RUN_NAME
#SBATCH -o ./logs/%x.%J.particle_load.out
#SBATCH -e ./logs/%x.%J.particle_load.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=1

module purge
module load intel_comp
module load openmpi
module load gnu_comp
module load parallel_hdf5
module load python/3.6.5

export OMPI_MCA_mpi_warn_on_fork=0

PL_INVOKE

ICGEN_SUBMIT

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode