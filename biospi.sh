#!/bin/bash
script="/mnt/storage/home/axel/Repository/PredictionOfHelminthsInfection/ml_pipeline.py"
out_dir="/mnt/storage/scratch/axel/new_pipeline/ml_output/"
in_dir="/mnt/storage/scratch/axel/new_pipeline/"

for dat in gain_1 gain_100 gain_raw li raw;do
        for cl in 2 4; do
                echo ${script} ${out_dir}ml_${dat}_1_${cl} ${in_dir}dataset_${dat} 1 ${cl} n n 6 > ml_${dat}_1${cl}.txt
  done
done