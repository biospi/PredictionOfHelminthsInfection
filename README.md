# Prediction Of Helminths Infection Pipeline (POHI)

This project process accelerometry data and uses machine learning to predict small ruminants(Goats and sheep) health.

## Project/Repo Structure
'project' code is found under projects/{projectname} with project names generally being named after the animal farm or study of interest. 'core' code is everything else.

## How To Use
```bash
usage: ml.py [-h] [--class_healthy CLASS_HEALTHY]
             [--class_unhealthy CLASS_UNHEALTHY] [--stratify STRATIFY]
             [--s_output S_OUTPUT] [--cwt CWT] [--n_scales N_SCALES]
             [--temp_file TEMP_FILE] [--hum_file HUM_FILE]
             [--n_splits N_SPLITS] [--n_repeats N_REPEATS] [--cv CV]
             [--wavelet_f0 WAVELET_F0] [--sfft_window SFFT_WINDOW]
             [--epochs EPOCHS] [--n_process N_PROCESS]
             output_dir dataset_folder

positional arguments:
  output_dir            Output directory
  dataset_folder        Dataset input directory

optional arguments:
  -h, --help            show this help message and exit
  --class_healthy CLASS_HEALTHY
                        Label for healthy class
  --class_unhealthy CLASS_UNHEALTHY
                        Label for unhealthy class
  --stratify STRATIFY   Enable stratiy for cross validation
  --s_output S_OUTPUT   Output sample files
  --cwt CWT             Enable freq domain (cwt)
  --n_scales N_SCALES   n scales in dyadic array [2^2....2^n].
  --temp_file TEMP_FILE
                        csv file containing temperature features.
  --hum_file HUM_FILE   csv file containing humidity features.
  --n_splits N_SPLITS   Number of splits for repeatedkfold cv
  --n_repeats N_REPEATS
                        Number of repeats for repeatedkfold cv
  --cv CV               Cross validation method (LeaveTwoOut|StratifiedLeaveTw
                        oOut|RepeatedStratifiedKFold|RepeatedKFold|LeaveOneOut
                        )
  --wavelet_f0 WAVELET_F0
                        Mother Wavelet frequency for CWT
  --sfft_window SFFT_WINDOW
                        STFT window size
  --epochs EPOCHS       Cnn epochs
  --n_process N_PROCESS
                        Number of threads to use.
```

