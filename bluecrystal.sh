#!/bin/env bash


#SBATCH --job-name=array_job
#SBATCH --partition=veryshort
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=100000M
#SBATCH --array=1-4

# Load the modules/environment
module purge
module load languages/anaconda3/3.7
conda init bash
conda activate /user/work/fo18103/PredictionOfHelminthsInfection/vgoat

# Define working directory
export WORK_DIR=/user/work/fo18103

# Change into working directory
cd ${WORK_DIR}

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo PBS ARRAY ID: ${SLURM_ARRAY_TASK_ID}
echo Working Directory: $(pwd)

# Execute code
python PredictionOfHelminthsInfection/ml.py --dataset-folder /user/work/fo18103/delmas/datasetraw_none_7day --n-imputed-days -1 --n-activity-days 1 --study-id delmas --preprocessing-steps LINEAR --cv RepeatedKFold --classifiers linear --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 1To2 --output-dir /user/work/fo18103/thesis/main_experiment/delmas_RepeatedKFold_-1_1_LINEAR_linear_rbf/1To1__1To2 > /user/work/fo18103/logs/${SLURM_JOBID}_RepeatedKFold_-1_1_LINEAR_linear_rbf__1To1__1To2.txt
python PredictionOfHelminthsInfection/ml.py --dataset-folder /user/work/fo18103/delmas/datasetraw_none_7day --n-imputed-days -1 --n-activity-days 1 --study-id delmas --preprocessing-steps LINEAR --preprocessing-steps QN --cv RepeatedKFold --classifiers linear --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 1To2 --output-dir /user/work/fo18103/thesis/main_experiment/delmas_RepeatedKFold_-1_1_LINEAR_QN_linear_rbf/1To1__1To2 > /user/work/fo18103/logs/${SLURM_JOBID}_RepeatedKFold_-1_1_LINEAR_QN_linear_rbf__1To1__1To2.txt
python PredictionOfHelminthsInfection/ml.py --dataset-folder /user/work/fo18103/cedara/datasetraw_none_7day --n-imputed-days -1 --n-activity-days 7 --study-id cedara --preprocessing-steps LINEAR --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 4To4 --class-unhealthy-label 3To5 --class-unhealthy-label 4To3 --class-unhealthy-label 5To3 --class-unhealthy-label 2To5 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/main_experiment/delmas_RepeatedKFold_-1_7_LINEAR_QN_ANSCOMBE_CENTER_CWTMORL_linear_rbf/1To1__4To4_3To5_4To3_5To3_2To5_2To2 > /user/work/fo18103/logs/${SLURM_JOBID}_RepeatedKFold_-1_7_LINEAR_QN_ANSCOMBE_CENTER_CWTMORL_linear_rbf__1To1__4To4_3To5_4To3_5To3_2To5_2To2.txt
python PredictionOfHelminthsInfection/ml.py --dataset-folder /user/work/fo18103/cedara/datasetraw_none_7day --n-imputed-days -1 --n-activity-days 7 --study-id cedara --preprocessing-steps LINEAR --preprocessing-steps QN --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 4To4 --class-unhealthy-label 3To5 --class-unhealthy-label 4To3 --class-unhealthy-label 5To3 --class-unhealthy-label 2To5 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/main_experiment/delmas_RepeatedKFold_-1_7_LINEAR_QN_LOG_CENTER_CWTMORL_linear_rbf/1To1__4To4_3To5_4To3_5To3_2To5_2To2 > /user/work/fo18103/logs/${SLURM_JOBID}_RepeatedKFold_-1_7_LINEAR_QN_LOG_CENTER_CWTMORL_linear_rbf__1To1__4To4_3To5_4To3_5To3_2To5_2To2.txt
