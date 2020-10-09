#!/bin/bash
njob = 26
for test_size in 40 25 10; do
  python ~/Repository/PredictionOfHelminthsInfection/pipeline/simplified_pipeline.py  ~/dev_simulation/FamachaMachineLearning/classification/classification_report/ ~/dev_simulation/new_pipeline/sweep/dataset_build_herd_scaled/**/*.data ${test_size} ${njob}
done
