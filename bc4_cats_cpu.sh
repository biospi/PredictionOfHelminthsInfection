#!/bin/env bash

#SBATCH --account=sscm012844
#SBATCH --job-name=cats_thesis
#SBATCH --output=cats_thesis
#SBATCH --error=cats_thesis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --mem=100000M
#SBATCH --array=1-6

# Load the modules/environment
module purge
module load languages/anaconda3/3.7
conda init
source ~/.bashrc


# Define working directory
export WORK_DIR=/user/work/fo18103/PredictionOfHelminthsInfection

# Change into working directory
cd ${WORK_DIR}
conda activate /user/work/fo18103/PredictionOfHelminthsInfection/vgoat

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo PBS ARRAY ID: ${SLURM_ARRAY_TASK_ID}
echo Working Directory: $(pwd)

cmds=('ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__006__0_00100__030/rbf/QN_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__006__0_00100__030/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf' 'ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__006__0_00100__060/rbf/QN_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__006__0_00100__060/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf' 'ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__005__0_00100__030/rbf/QN_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__005__0_00100__030/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf' 'ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__006__0_00100__030/rbf/QN_STD_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__006__0_00100__030/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --preprocessing-steps STD --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf' 'ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__006__0_00100__060/rbf/QN_STD_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__006__0_00100__060/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --preprocessing-steps STD --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf' 'ml.py --c None --gamma None --study-id cat --output-dir E:/Cats/ml_build_permutations_thesis_rev/1000__005__0_00100__030/rbf/QN_STD_LeaveOneOut --dataset-folder E:/Cats/build_permutations_final/1000__005__0_00100__030/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --syhth-thresh 4 --n-weather-days 7 --weather-file . --n-job 28 --cv LeaveOneOut --preprocessing-steps QN --preprocessing-steps STD --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf')
# Execute code
echo ${cmds[${SLURM_ARRAY_TASK_ID}]}
python ${cmds[${SLURM_ARRAY_TASK_ID}]} > /user/work/fo18103/logs/cats_thesis_${SLURM_ARRAY_TASK_ID}.log
