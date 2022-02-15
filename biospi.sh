#!/bin/bash
script="/mnt/storage/home/axel/Repository/PredictionOfHelminthsInfection/ml_pipeline.py"
out_dir="/mnt/storage/scratch/axel/new_pipeline/ml_output/"
in_dir="/mnt/storage/scratch/axel/new_pipeline/"

for dat in gain_1 gain_100 gain_raw li raw;do
        for cl in 2 4; do
                echo ${script} ${out_dir}ml_${dat}_1_${cl} ${in_dir}dataset_${dat} 1 ${cl} n n 6 > ml_${dat}_1${cl}.txt
  done
done

nohup python3 ml.py /mnt/storage/scratch/axel/cats/ml/ml_sec/day_w /mnt/storage/scratch/axel/cats/build_sec/dataset/training_sets/day_w --preprocessing-steps [["LINEAR", "QN", "ANSCOMBE", "LOG"]] --meta-columns ["label", "id", "imputed_days", "date", "health", "target", "age", "name", "mobility_score"] --n-imputed-days -1 --n-activity-days -1 --class-healthy-label ['0.0'] --class-unhealthy-label ['1.0'] --n-splits 5 --n-repeats 10 --n-job 30 > log.txt &

nohup python3 ml.py /mnt/storage/scratch/axel/cats/ml/ml_sec/day_w /mnt/storage/scratch/axel/cats/build_sec/dataset/training_sets/day_w --preprocessing-steps [["LINEAR", "QN", "ANSCOMBE", "LOG"]] > log.txt &