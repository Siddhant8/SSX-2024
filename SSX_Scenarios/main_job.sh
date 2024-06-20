#!/bin/bash
# JOB HEADERS HERE

#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 14:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 64
#SBATCH --mail-user=askeldo1@swarthmore.edu
#SBATCH --mail-type=ALL
#echo commands to stdout
set -x

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

#activate dedalus
source /jet/home/skeldon/anaconda3/bin/activate 
conda activate base 
conda activate dedalus3


#run mpi program
cd /ocean/projects/phy190003p/skeldon/month_of_9_23

mpirun -np $SLURM_NTASKS python3 D3_IVP_SSX.py
