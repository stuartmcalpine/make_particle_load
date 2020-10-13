#!/bin/bash -l

#SBATCH --ntasks=112
#SBATCH -J FILENAME
#SBATCH -o ./logs/%x.%J.particleload.out
#SBATCH -e ./logs/%x.%J.particleload.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1

export OMPI_MCA_mpi_warn_on_fork=0

module purge
module load gnu_comp
module load openmpi
module load parallel_hdf5
module load python/3.6.5

#mpirun -np $SLURM_NTASKS python3 submit_from_list.py \
# -t param_files/default_sept.yml \
# -l /cosma7/data/dp004/dc-alta2/xl-zooms/ics/masks/groupnumbers_defaultSept.txt \
# -s

mpirun -np $SLURM_NTASKS python3 Generate_PL.py \
 param_files/default_sept.yml

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode