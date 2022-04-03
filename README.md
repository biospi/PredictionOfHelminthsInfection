# Prediction Of Helminths Infection Pipeline (POHI)

This project process accelerometry data and uses machine learning to predict small ruminants(Goats and sheep) health.

## Project/Repo Structure
'project' code is found under projects/{projectname} with project names generally being named after the animal farm or study of interest. 'core' code is everything else.

## How To Use
```bash
Usage: ml.py [OPTIONS]

  ML

  Args:
      output_dir: Output directory
      dataset_folder: Dataset input directory
      class_healthy: Label for healthy class
      class_unhealthy: Label for unhealthy class
      stratify: Enable stratiy for cross validation
      s_output: Output sample files
      cwt: Enable freq domain (cwt)
      n_scales: n scales in dyadic array [2^2....2^n].
      temp_file: csv file containing temperature features.
      hum_file: csv file containing humidity features.
      n_splits: Number of splits for repeatedkfold cv.
      n_repeats: Number of repeats for repeatedkfold cv.
      cv: RepeatedKFold
      wavelet_f0: Mother Wavelet frequency for CWT
      sfft_window: STFT window size
      epochs: Cnn epochs
      n_process:Number of threads to use.

Options:
  --output-dir DIRECTORY          [required]
  --dataset-folder DIRECTORY      [required]
  --preprocessing-steps TEXT      [default: QN, ANSCOMBE, LOG, DIFF]
  --class-healthy-label TEXT      [default: 1To1]
  --class-unhealthy-label TEXT    [default: 1To2]
  --stratify TEXT                 [default: n]
  --n-scales INTEGER              [default: 30]
  --hum-file PATH                 [default: .]
  --temp-file PATH                [default: .]
  --n-splits INTEGER              [default: 5]
  --n-repeats INTEGER             [default: 10]
  --epochs INTEGER                [default: 20]
  --n-process INTEGER             [default: 6]
  --output-samples / --no-output-samples
                                  [default: True]
  --output-cwt / --no-output-cwt  [default: True]
  --cv TEXT                       [default: RepeatedKFold]
  --wavelet-f0 INTEGER            [default: 6]
  --sfft-window INTEGER           [default: 60]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```

##Blue Crystal 4
```bash
module load tools/git/2.18.0
module load languages/anaconda3/2020-3.8.5
conda create --prefix /user/work/fo18103/PredictionOfHelminthsInfection/vgoat python=3.8.5
conda activate /user/work/fo18103/PredictionOfHelminthsInfection/vgoat
export PATH=/user/work/fo18103/PredictionOfHelminthsInfection/vgoat/bin/:$PATH
python -m pip install --upgrade pip
make environment


languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch
module load Anaconda3/2019.10
conda create -n {envname} python=3.7.2
conda activate {envname}
module load libvips/8.9.2-foss-2019a
export PATH=/user/work/fo18103/PredictionOfHelminthsInfection/vgoat/bin/:$PATH
export JAVA_HOME=/usr/java/jdk1.8.0_112/
make environment
```