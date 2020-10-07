#!/bin/bash
#rm -rf /home/axel/dev_simulation/new_pipeline/thresholded/
njob=24 
for threshz in `seq 240 120 720`;do
	for threshi in 3 5 10 15 20 30 45 60 80 100 120; do
		python interpolation_zero2nan_thresh.py /home/axel/dev_simulation/new_pipeline/thresholded /home/axel/dev_simulation/new_pipeline/backfill_1min/delmas_70101200027/ $threshz $threshi $njob
	done
done

