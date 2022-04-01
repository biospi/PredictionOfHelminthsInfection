#!/bin/env bash


#SBATCH --job-name=array_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=28
#SBATCH --time=1:0:0
#SBATCH --mem=100000M
#SBATCH --array=1-1000

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate goat

# Define working directory
export WORK_DIR=${HOME}/PredictionOfHelminthsInfection


# Change into working directory
cd ${WORK_DIR}

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo PBS ARRAY ID: ${SLURM_ARRAY_TASK_ID}
echo Working Directory: $(pwd)

# Execute code
python ml.py > ${SLURM_ARRAY_TASK_ID}.txt