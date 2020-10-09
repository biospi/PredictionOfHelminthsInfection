#!/bin/bash
for threshz in `seq 240 120 720`;do
        for threshi in 3 5 10 15 20 30 45 60 80 100 120; do
		python create_median_animal.py /home/axel/dev_simulation/new_pipeline/thresholded/delmas_70101200027/interpol_${threshi}_zeros_${threshz}/*.csv
        done
done

