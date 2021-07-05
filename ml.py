#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#

#%%

# %%
import argparse
import glob
import os

import pandas as pd

from model.data_loader import loadActivityData, parse_param_from_filename
from model.svm import process_data_frame_svm
from preprocessing.preprocessing import applyPreprocessingSteps
from utils.visualisation import plotMlReport, plotMeanGroups, \
    plot_zeros_distrib, plot_groups, plot_time_lda, plot_time_pca, plotHeatmap



def main(preprocessing_steps, output_dir, dataset_folder, class_healthy_label, class_unhealthy_label, stratify, n_scales,
         hum_file, temp_file, n_splits, n_repeats, epochs, n_process, output_samples, output_cwt, cv, wavelet_f0, sfft_window):
    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_folder)
    print("class_healthy=", class_healthy_label)
    print("class_unhealthy=", class_unhealthy_label)
    print("output_samples=", output_samples)
    print("stratify=", stratify)
    print("output_cwt=", output_cwt)
    print("hum_file=", hum_file)
    print("temp_file=", temp_file)
    print("epochs=", epochs)
    print("n_process=", n_process)
    print("output_samples=", output_samples)
    print("output_cwt=", output_cwt)
    print("cv=", cv)
    print("wavelet_f0=", wavelet_f0)
    print("sfft_window=", sfft_window)
    print("loading dataset...")
    enable_downsample_df = False
    day = int([a for a in dataset_folder.split('_') if "day" in a][0][0])

    files = glob.glob(dataset_folder + "/*.csv")  # find datset files
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))
    print(files)

    has_humidity_data = False
    df_hum = None
    if hum_file is not None:
        has_humidity_data = True
        print("humidity file detected!", hum_file)
        df_hum = pd.read_csv(hum_file)
        print(df_hum.shape)
        plotHeatmap(df_hum.values, output_dir, "Samples humidity", "humidity.html")

    has_temperature_data = True
    df_temp = None
    if temp_file is not None:
        has_temperature_data = True
        print("temperature file detected!", temp_file)
        df_temp = pd.read_csv(temp_file)
        plotHeatmap(df_temp.values, output_dir, "Samples temperature", "temperature.html")
        print(df_temp.shape)

    has_humidity_and_temp = False
    df_hum_temp = None
    if temp_file is not None and hum_file is not None:
        has_humidity_and_temp = True
        print("temperature file detected!", temp_file)
        print("humidity file detected!", hum_file)
        df_hum_temp = pd.concat([df_temp, df_hum], axis=1)
        plotHeatmap(df_hum_temp.values, output_dir, "Samples temperature and Humidity", "temperature_humidity.html")
        print(df_hum_temp.shape)

    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print("loading dataset file %s ..." % file)
        data_frame, N_META, class_healthy_target, class_unhealthy_target, label_series = loadActivityData(file, day,
                                                                                                          class_healthy_label, class_unhealthy_label)

        #plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
        ################################################################################################################
        ##VISUALISATION
        ################################################################################################################
        animal_ids = data_frame.iloc[0:len(data_frame), :]["id"].astype(str).tolist()
        df_norm = applyPreprocessingSteps(days, df_hum, df_temp, sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, ["QN"],
                                          class_healthy_label, class_unhealthy_label, class_healthy_target, class_unhealthy_target,
                                          clf_name="SVM_QN_VISU", n_scales=n_scales)
        plot_zeros_distrib(label_series, df_norm, output_dir,
                           title='Percentage of zeros in activity per sample after normalisation')
        plot_zeros_distrib(label_series, data_frame.copy(), output_dir,
                           title='Percentage of zeros in activity per sample before normalisation')
        plotMeanGroups(n_scales, sfft_window, wavelet_f0, df_norm, label_series, N_META, output_dir + "/raw_after_qn/")

        plot_time_pca(N_META, data_frame.copy(), output_dir, label_series, title="PCA time domain before normalisation")
        plot_time_pca(N_META, df_norm, output_dir, label_series, title="PCA time domain after normalisation")

        plot_time_lda(N_META, data_frame.copy(), output_dir, label_series, title="LDA time domain before normalisation")
        plot_time_lda(N_META, data_frame.copy(), output_dir, label_series, title="LDA time domain after normalisation")

        ntraces = 2
        idx_healthy, idx_unhealthy = plot_groups(N_META, animal_ids, class_healthy_label, class_unhealthy_label,
                                                 class_healthy_target, class_unhealthy_target, output_dir, data_frame.copy(), title="Raw imputed",
                                                 xlabel="Time",
                                                 ylabel="activity", ntraces=ntraces)
        plot_groups(N_META, animal_ids, class_healthy_label, class_unhealthy_label, class_healthy_target, class_unhealthy_target,
                    output_dir,
                    df_norm, title="Normalised(Quotient Norm) samples", xlabel="Time", ylabel="activity",
                    idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=2, ntraces=ntraces)
        ################################################################################################################
        # keep only two class of samples
        data_frame = data_frame[data_frame["target"].isin([class_healthy_target, class_unhealthy_target])]
        animal_ids = data_frame.iloc[0:len(data_frame), :]["id"].astype(str).tolist()
        # cv = "StratifiedLeaveTwoOut"

        for steps in preprocessing_steps:
            step_slug = "_".join(steps)
            df_processed = applyPreprocessingSteps(days, df_hum, df_temp, sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, steps,
                                                   class_healthy_label, class_unhealthy_label, class_healthy_target,
                                                   class_unhealthy_target, clf_name="SVM", output_dim=data_frame.shape[0],
                                                   n_scales=n_scales)
            # targets = df_processed["target"]
            # df_processed = df_processed.iloc[:, :-N_META]
            # df_processed["target"] = targets
            process_data_frame_svm(output_dir, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug,
                                   n_splits, n_repeats,
                                   sampling, enable_downsample_df, label_series, class_healthy_target, class_unhealthy_target,
                                   cv=cv)
        #2DCNN
        # for steps in [["QN", "ANSCOMBE", "LOG"]]:
        #     step_slug = "_".join(steps)
        #     step_slug = step_slug + "_2DCNN"
        #     df_processed = applyPreprocessingSteps(data_frame.copy(), N_META, output_dir, steps,
        #                                            class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, clf_name="2DCNN")
        #     targets = df_processed["target"]
        #     df_processed = df_processed.iloc[:, :-N_META]
        #     df_processed["target"] = targets
        #     process_data_frame_2dcnn(epochs, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug, n_splits, n_repeats, sampling,
        #                    enable_downsample_df, label_series, class_healthy, class_unhealthy, cv="StratifiedLeaveTwoOut")

        #1DCNN
        # for steps in [["QN"], ["QN", "ANSCOMBE", "LOG"]]:
        #     step_slug = "_".join(steps)
        #     step_slug = step_slug + "_1DCNN"
        #     df_processed = applyPreprocessingSteps(sfft_window, wavelet_f0, animal_ids, data_frame.copy(), N_META, output_dir, steps,
        #                                            class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, clf_name="1DCNN", output_dim=data_frame.shape[0])
        #     targets = df_processed["target"]
        #     df_processed = df_processed.iloc[:, :-N_META]
        #     df_processed["target"] = targets
        #     process_data_frame_1dcnn(epochs, stratify, animal_ids, output_dir, df_processed, days, farm_id, step_slug, n_splits, n_repeats, sampling,
        #                    enable_downsample_df, label_series, class_healthy, class_unhealthy, cv=cv)

    output_dir = "%s/%s" % (output_dir, cv)
    files = [output_dir + "/" + file for file in os.listdir(output_dir) if file.endswith(".csv")]
    print("found %d files." % len(files))
    print("compiling final file...")
    df_final = pd.DataFrame()
    dfs = [pd.read_csv(file, sep=",") for file in files]
    df_final = pd.concat(dfs)
    filename = "%s/final_classification_report.csv" % output_dir
    df_final.to_csv(filename, sep=',', index=False)
    print(df_final)
    plotMlReport(filename, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Output directory', type=str)
    parser.add_argument('dataset_folder', help='Dataset input directory', type=str)
    parser.add_argument('--class_healthy', help='Label for healthy class', default="1To1", type=str)
    parser.add_argument('--class_unhealthy', help='Label for unhealthy class', default="1To2", type=str)
    parser.add_argument('--stratify', help='Enable stratiy for cross validation', default='n', type=str)
    parser.add_argument('--s_output', help='Output sample files', default='y', type=str)
    parser.add_argument('--cwt', help='Enable freq domain (cwt)', default='y', type=str)
    parser.add_argument('--n_scales', help='n scales in dyadic array [2^2....2^n].', default=10, type=int)
    parser.add_argument('--temp_file', help='csv file containing temperature features.', default=None, type=str)
    parser.add_argument('--hum_file', help='csv file containing humidity features.', default=None, type=str)
    parser.add_argument('--n_splits', help='Number of splits for repeatedkfold cv', default=5, type=int)
    parser.add_argument('--n_repeats', help='Number of repeats for repeatedkfold cv', default=10, type=int)
    parser.add_argument('--cv', help='Cross validation method (LeaveTwoOut|StratifiedLeaveTwoOut|RepeatedStratifiedKFold|RepeatedKFold|LeaveOneOut)',
                        default="RepeatedKFold", type=str)
    parser.add_argument('--wavelet_f0', help='Mother Wavelet frequency for CWT', default=6, type=int)
    parser.add_argument('--sfft_window', help='STFT window size', default=60, type=int)
    parser.add_argument('--epochs', help='Cnn epochs', default=20, type=int)
    parser.add_argument('--n_process', help='Number of threads to use.', default=6, type=int)

    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_folder = args.dataset_folder
    class_healthy = args.class_healthy
    class_unhealthy = args.class_unhealthy
    stratify = args.stratify
    s_output = args.s_output
    cwt = args.cwt
    n_scales = args.n_scales
    hum_file = args.hum_file
    temp_file = args.temp_file
    n_splits = args.n_splits
    n_repeats = args.n_repeats
    epochs = args.epochs
    n_process = args.n_process
    cv = args.cv
    wavelet_f0 = args.wavelet_f0
    sfft_window = args.sfft_window

    stratify = "y" in stratify.lower()
    output_samples = "y" in s_output.lower()
    output_cwt = "y" in cwt.lower()

    steps = [
             ["QN", "ANSCOMBE", "LOG", "DIFF"]
             # ["QN", "ANSCOMBE", "LOG", "DIFF", "UMAP"],
             # ["QN", "ANSCOMBE", "LOG", "DIFF", "STANDARDSCALER", "UMAP"],
             # ["QN", "ANSCOMBE", "LOG"],
             # ["QN", "ANSCOMBE", "LOG", "DIFFLASTD"],
             # ["QN", "ANSCOMBE", "LOG", "DIFF", "STANDARDSCALER"],
             # ["QN", "ANSCOMBE", "LOG", "DIFF", "CWT(MORL)", "UMAP"],
             # ["QN", "ANSCOMBE", "LOG", "DIFF", "CWT(MORL)", "STANDARDSCALER", "UMAP"]
             ]

    main(steps, output_dir, dataset_folder, class_healthy, class_unhealthy, stratify, n_scales,
         hum_file, temp_file, n_splits, n_repeats, epochs, n_process, output_samples, output_cwt, cv, wavelet_f0, sfft_window)