#!/bin/env bash


#SBATCH --job-name=array_job_cats
#SBATCH --output=cat_job_out
#SBATCH --error=cat_job_error
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2-00:00:00
#SBATCH --mem=480000M
#SBATCH --array=1-16

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


cmds=('ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/008__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/008__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/008__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/008__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/008__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/008__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/008__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/008__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/004__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/004__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/003__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/003__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/002__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/002__0_00100__120/cats_LeaveOneOut_-1_-1_QN_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/002__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps STD --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/002__0_00100__120/cats_LeaveOneOut_-1_-1_QN_STD_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/002__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/002__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_linear_rbf/0.0__1.0' 'ml.py --dataset-folder /user/work/fo18103/cats_data/build_multiple_peak_permutations/002__0_00100__120/dataset/training_sets/samples --n-imputed-days -1 --n-activity-days -1 --study-id cats --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --meta-columns label --meta-columns id --meta-columns imputed_days --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --cv LeaveOneOut --classifiers linear --classifiers rbf --class-healthy-label 0.0 --class-unhealthy-label 1.0 --output-dir /user/work/fo18103/cats_data/ml_build_multiple_peak_permutations/002__0_00100__120/cats_LeaveOneOut_-1_-1_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear_rbf/0.0__1.0'
)

# Execute code
echo ${cmds[${SLURM_ARRAY_TASK_ID}]}
python ${cmds[${SLURM_ARRAY_TASK_ID}]} > /user/work/fo18103/logs/cats_${SLURM_ARRAY_TASK_ID}.log
