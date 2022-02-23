#!/bin/bash
script='/mnt/storage/home/axel/Repository/PredictionOfHelminthsInfection/ml_pipeline.py'
out_dir='/mnt/storage/scratch/axel/new_pipeline/ml_output/'
in_dir='/mnt/storage/scratch/axel/new_pipeline/'

for dat in gain_1 gain_100 gain_raw li raw;do
        for cl in 2 4; do
                echo ${script} ${out_dir}ml_${dat}_1_${cl} ${in_dir}dataset_${dat} 1 ${cl} n n 6 > ml_${dat}_1${cl}.txt
  done
done

nohup python3 ml.py --output-dir /mnt/storage/scratch/axel/cats/ml/ml_min/day_w --dataset-folder /mnt/storage/scratch/axel/cats/build_min/dataset/training_sets/day_w --preprocessing-steps 'LINEAR' --preprocessing-steps 'QN' --preprocessing-steps 'ANSCOMBE' --preprocessing-steps 'LOG' --preprocessing-steps 'CENTER' --preprocessing-steps 'CWT' --meta-columns 'label' --meta-columns 'id' --meta-columns 'imputed_days' --meta-columns 'date' --meta-columns 'health' --meta-columns 'target' --meta-columns 'age' --meta-columns 'name' --meta-columns 'mobility_score' --n-imputed-days -1 --n-activity-days -1 --class-healthy-label '0.0' --class-unhealthy-label '1.0' --n-splits 5 --n-repeats 5 --n-job 30 > log.txt &


nohup python3 ml.py /mnt/storage/scratch/axel/cats/ml/ml_sec/day_w /mnt/storage/scratch/axel/cats/build_sec/dataset/training_sets/day_w --preprocessing-steps [['LINEAR', 'QN', 'ANSCOMBE', 'LOG']] > log.txt &


nohup python3 ml.py --output-dir /mnt/storage/scratch/axel/thesis/main/delmas/2To2 --dataset-folder /mnt/storage/scratch/axel/dataset4_mrnn_7day --preprocessing-steps 'QN' --preprocessing-steps 'ANSCOMBE' --preprocessing-steps 'LOG' --preprocessing-steps 'CENTER' --preprocessing-steps 'CWT' --meta-columns 'label' --meta-columns 'id' --meta-columns 'imputed_days' --meta-columns 'date' --meta-columns 'health' --meta-columns 'target' --n-imputed-days 7 --n-activity-days 7 --class-healthy-label '1To1' --class-unhealthy-label '2To2' --n-splits 5 --n-repeats 10 --n-job 25 > log_delmas.txt &
