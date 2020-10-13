#!/bin/bash -l

#SBATCH --ntasks=XXX
#SBATCH --cpus-per-task=28
#SBATCH -J XXX
#SBATCH -o ./logs/%x.%J.swift.out
#SBATCH -e ./logs/%x.%J.swift.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --time=XXX:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load ucx/1.8.1
module load parmetis/4.0.3-64bit
module load parallel_hdf5/1.10.6
module load fftw/3.3.8cosma7
module load gsl/2.5


# Make a dmo/hydro switch
if [[ $PWD == *"hydro"* ]]; then
  # If this file in "/hydro" subdirectory, use hydro SWIFT and hydro parameter file
  /cosma7/data/dp004/dc-alta2/xl-zooms/hydro/swiftsim/examples/swift \
  -v 1 \
  --cosmology \
  --eagle \
  --pin \
  --threads=$SLURM_CPUS_PER_TASK \
  ./config/params.yml
fi

if [[ $PWD == *"dmo"* ]]; then
  # If this file in "/dmo" subdirectory, use dmo SWIFT and dmo parameter file
  /cosma7/data/dp004/dc-alta2/xl-zooms/dmo/swiftsim/examples/swift \
  -v 1 \
  --cosmology \
  --self-gravity \
  --pin \
  --threads=$SLURM_CPUS_PER_TASK \
  ./config/params.yml
fi

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode