#!/bin/env bash


#SBATCH --job-name=array_job
#SBATCH --partition=short
#SBATCH --output=cat_job_out
#SBATCH --error=cat_job_error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --mem=100000M
#SBATCH --array=1-8

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2019.03-tensorflow
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

cmds=('ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_CENTER_CWTMORL_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_CENTER_CWTMORL_linear_rbf/0.0__1.0'
)

# Execute code
echo ${cmds[${SLURM_ARRAY_TASK_ID}]}
python ${cmds[${SLURM_ARRAY_TASK_ID}]} > /user/work/fo18103/logs/cats_${SLURM_ARRAY_TASK_ID}.log
