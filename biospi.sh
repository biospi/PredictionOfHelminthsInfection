#!/bin/bash
script='/mnt/storage/home/axel/Repository/PredictionOfHelminthsInfection/ml_pipeline.py'
out_dir='/mnt/storage/scratch/axel/new_pipeline/ml_output/'
in_dir='/mnt/storage/scratch/axel/new_pipeline/'

for dat in gain_1 gain_100 gain_raw li raw;do
        for cl in 2 4; do
                echo ${script} ${out_dir}ml_${dat}_1_${cl} ${in_dir}dataset_${dat} 1 ${cl} n n 6 > ml_${dat}_1${cl}.txt
  done
done

nohup python3 ml.py --output-dir /mnt/storage/scratch/axel/cats/ml/ml_1440_1440/day_w --dataset-folder /mnt/storage/scratch/axel/cats/build_min_1440_1440/dataset/training_sets/day_w --preprocessing-steps 'LINEAR' --preprocessing-steps 'QN' --preprocessing-steps 'ANSCOMBE' --preprocessing-steps 'LOG' --meta-columns 'label' --meta-columns 'id' --meta-columns 'imputed_days' --meta-columns 'date' --meta-columns 'health' --meta-columns 'target' --meta-columns 'age' --meta-columns 'name' --meta-columns 'mobility_score' --n-imputed-days -1 --n-activity-days -1 --class-healthy-label '0.0' --class-unhealthy-label '1.0' --classifiers 'cnn' --n-splits 5 --n-repeats 10 --n-job 1 > log.txt &


nohup python3 ml.py /mnt/storage/scratch/axel/cats/ml/ml_sec/day_w /mnt/storage/scratch/axel/cats/build_sec/dataset/training_sets/day_w --preprocessing-steps [['LINEAR', 'QN', 'ANSCOMBE', 'LOG']] > log.txt &


nohup python3 ml.py --output-dir /mnt/storage/scratch/axel/thesis2/main/delmas/2To2 --dataset-folder /mnt/storage/scratch/axel/dataset4_mrnn_7day --preprocessing-steps 'QN' --preprocessing-steps 'ANSCOMBE' --preprocessing-steps 'LOG' --preprocessing-steps 'CENTER' --preprocessing-steps 'CWT' --meta-columns 'label' --meta-columns 'id' --meta-columns 'imputed_days' --meta-columns 'date' --meta-columns 'health' --meta-columns 'target' --n-imputed-days 7 --n-activity-days 7 --class-healthy-label '1To1' --class-unhealthy-label '2To2' --classifiers 'cnn' --n-splits 5 --n-repeats 10 --n-job 6 --batch-size 8 > log_delmas.txt &


nohup python3 run_thesis.py --output-dir /mnt/storage/scratch/axel/thesis3 --delmas-dir /mnt/storage/scratch/axel/dataset4_mrnn_7day --cedara-dir /mnt/storage/scratch/axel/dataset6_mrnn_7day