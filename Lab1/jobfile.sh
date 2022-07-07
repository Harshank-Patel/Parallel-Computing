#!/bin/bash
### Job commands start here
## echo '=====================JOB STARTING=========================='
#SBATCH --job-name=openmpcode              ### Job Name
#SBATCH --output=output.%j        ### File in which to store job output
#SBATCH --error=error.%j          ### File in which to store job error messages
#SBATCH --time=0-00:15:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job, default = 1
#SBATCH --ntasks-per-node=1     ### Nuber of tasks to be launched per Node, default = 1
#SBATCH --mem=8G                ### 8GB memory per node
#SBATCH --cpus-per-task=48  ### Number of threads per task (OMP threads) and set SLURM_CPUS_PER_TASK
#SBATCH --exclusive             ### no shared resources within a node
export OMP_PROC_BIND='close'
echo ' thread affinity/proc_bind = ' ; echo $OMP_PROC_BIND
./luDecomposition "$@"
### Display some diagnostic information
echo '=====================JOB DIAGNOTICS========================'
date
echo -n 'This machine is ';hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo 'My path is:' 
echo $PATH
sinfo -s
echo 'My job info:'
squeue -j $SLURM_JOBID
echo 'Machine info'
echo ' '
echo '========================ALL DONE==========================='