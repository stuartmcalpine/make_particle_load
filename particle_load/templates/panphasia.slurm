#!/bin/bash -l

#SBATCH --ntasks=XXX
#SBATCH --job-name=XXX
#SBATCH --output=out_files/XXX.log
#SBATCH --partition=cosma6
#SBATCH --account=dp004
#SBATCH --exclusive
#SBATCH --time=XXX:00:00

module purge
module load intel_comp/2018
module load intel_mpi/2018
module load fftw/2.1.5
module load gsl/2.4
module load hdf5/1.10.3

cd ..

mpirun -env I_MPI_PIN=1 -env I_MPI_PIN_PROCESSOR_LIST=allcores -np $SLURM_NTASKS ./IC_Gen.x Param_Files/XXX.inp

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

