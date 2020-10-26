
#!/bin/bash
njob=20
for test_size in 40 25 10; do
	python ~/Repository/PredictionOfHelminthsInfection/pipeline/ml_pipeline.py  ~/dev_simulation/FamachaMachineLearning/classification/classification_report/ ~/dev_simulation/new_pipeline/sweep/dataset_build/**/*.data $test_size $njob
done
