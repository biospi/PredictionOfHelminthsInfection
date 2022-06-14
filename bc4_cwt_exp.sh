#!/bin/env bash


#SBATCH --job-name=cwt
#SBATCH --output=cwt_job_out
#SBATCH --error=cwt_job_error
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2-00:00:00
#SBATCH --mem=480000M
#SBATCH --array=1-28

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

cmds=('ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps DWT --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_DWT_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --cv RepeatedKFold --classifiers linear --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_linear/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 6 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 12 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 1 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 18 --sub-sample-scales 6 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps CWTMORL --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_CWTMORL_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --preprocessing-steps CENTER --preprocessing-steps DWT --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_CENTER_DWT_rbf/1To1__2To2' 'ml.py --dataset-folder /user/work/fo18103/delmas/dataset4_mrnn_7day --n-imputed-days 3 --n-activity-days 3 --study-id delmas --n-scales 9 --sub-sample-scales 3 --preprocessing-steps QN --preprocessing-steps ANSCOMBE --preprocessing-steps LOG --cv RepeatedKFold --classifiers rbf --class-healthy-label 1To1 --class-unhealthy-label 2To2 --output-dir /user/work/fo18103/thesis/cwt_optimal_exp/delmas_RepeatedKFold_3_3_QN_ANSCOMBE_LOG_rbf/1To1__2To2'
)

# Execute code
echo ${cmds[${SLURM_ARRAY_TASK_ID}]}
python ${cmds[${SLURM_ARRAY_TASK_ID}]} > /user/work/fo18103/logs/cwt_${SLURM_ARRAY_TASK_ID}.log