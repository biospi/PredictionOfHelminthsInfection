#!/bin/bash
#rm -rf /home/axel/dev_simulation/new_pipeline/thresholded/
njob=24 
threshz=480
for threshi in 1; do
  python interpolation_zero2nan_thresh.py /home/axel/dev_simulation/new_pipeline/thresholded /home/axel/dev_simulation/new_pipeline/backfill_1min/delmas_70101200027/ $threshz $threshi $njob
done


