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

import glob
import os
from pathlib import Path
from typing import List, Optional

import typer
import pandas as pd

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import process_data_frame_svm
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import (
    plotMlReport,
    plotMeanGroups,
    plot_zeros_distrib,
    plot_groups,
    plot_time_lda,
    plot_time_pca,
    plotHeatmap,
)


def main(
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    dataset_folder: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    preprocessing_steps: List[str] = [["QN", "ANSCOMBE", "LOG", "DIFF"]],
    class_healthy_label: str = "1To1",
    class_unhealthy_label: str = "2To2",
    n_scales: int = 30,
    hum_file: Optional[Path] = Path('.'),
    temp_file: Optional[Path] = Path('.'),
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedKFold",
    wavelet_f0: int = 6,
    sfft_window: int = 60,
):
    """ML\n
    Args:\n
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
    """

    enable_downsample_df = False
    day = int([a for a in dataset_folder.name.split("_") if "day" in a][0][0])

    files = glob.glob(str(dataset_folder / "*.csv"))  # find datset files
    print("found %d files." % len(files))
    print(files)

    df_hum = None
    if hum_file.is_file():
        print("humidity file detected!", hum_file)
        df_hum = pd.read_csv(hum_file)
        print(df_hum.shape)
        plotHeatmap(df_hum.values, output_dir, "Samples humidity", "humidity.html")

    df_temp = None
    if temp_file.is_file():
        print("temperature file detected!", temp_file)
        df_temp = pd.read_csv(temp_file)
        plotHeatmap(
            df_temp.values, output_dir, "Samples temperature", "temperature.html"
        )
        print(df_temp.shape)

    df_hum_temp = None
    if temp_file.is_file() and hum_file.is_file():
        print("temperature file detected!", temp_file)
        print("humidity file detected!", hum_file)
        df_hum_temp = pd.concat([df_temp, df_hum], axis=1)
        plotHeatmap(
            df_hum_temp.values,
            output_dir,
            "Samples temperature and Humidity",
            "temperature_humidity.html",
        )
        print(df_hum_temp.shape)

    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            N_META,
            class_healthy_target,
            class_unhealthy_target,
            label_series,
        ) = load_activity_data(file, day, class_healthy_label, class_unhealthy_label)

        # plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
        ###############
        #VISUALISATION
        ###############
        animal_ids = data_frame.iloc[0: len(data_frame), :]["id"].astype(str).tolist()
        df_norm = apply_preprocessing_steps(
            days,
            df_hum,
            df_temp,
            sfft_window,
            wavelet_f0,
            animal_ids,
            data_frame.copy(),
            N_META,
            output_dir,
            ["QN"],
            class_healthy_label,
            class_unhealthy_label,
            class_healthy_target,
            class_unhealthy_target,
            clf_name="SVM_QN_VISU",
            n_scales=n_scales,
        )
        plot_zeros_distrib(
            label_series,
            df_norm,
            output_dir,
            title="Percentage of zeros in activity per sample after normalisation",
        )
        plot_zeros_distrib(
            label_series,
            data_frame.copy(),
            output_dir,
            title="Percentage of zeros in activity per sample before normalisation",
        )
        plotMeanGroups(
            n_scales,
            sfft_window,
            wavelet_f0,
            df_norm,
            label_series,
            N_META,
            output_dir / "raw_after_qn",
        )

        plot_time_pca(
            N_META,
            data_frame.copy(),
            output_dir,
            label_series,
            title="PCA time domain before normalisation",
        )
        plot_time_pca(
            N_META,
            df_norm,
            output_dir,
            label_series,
            title="PCA time domain after normalisation",
        )

        plot_time_lda(
            N_META,
            data_frame.copy(),
            output_dir,
            label_series,
            title="LDA time domain before normalisation",
        )
        plot_time_lda(
            N_META,
            data_frame.copy(),
            output_dir,
            label_series,
            title="LDA time domain after normalisation",
        )

        ntraces = 2
        idx_healthy, idx_unhealthy = plot_groups(
            N_META,
            animal_ids,
            class_healthy_label,
            class_unhealthy_label,
            class_healthy_target,
            class_unhealthy_target,
            output_dir,
            data_frame.copy(),
            title="Raw imputed",
            xlabel="Time",
            ylabel="activity",
            ntraces=ntraces,
        )
        plot_groups(
            N_META,
            animal_ids,
            class_healthy_label,
            class_unhealthy_label,
            class_healthy_target,
            class_unhealthy_target,
            output_dir,
            df_norm,
            title="Normalised(Quotient Norm) samples",
            xlabel="Time",
            ylabel="activity",
            idx_healthy=idx_healthy,
            idx_unhealthy=idx_unhealthy,
            stepid=2,
            ntraces=ntraces,
        )
        ################################################################################################################
        # keep only two class of samples
        data_frame = data_frame[
            data_frame["target"].isin([class_healthy_target, class_unhealthy_target])
        ]
        animal_ids = data_frame.iloc[0: len(data_frame), :]["id"].astype(str).tolist()
        # cv = "StratifiedLeaveTwoOut"

        for steps in preprocessing_steps:
            step_slug = "_".join(steps)
            df_processed = apply_preprocessing_steps(
                days,
                df_hum,
                df_temp,
                sfft_window,
                wavelet_f0,
                animal_ids,
                data_frame.copy(),
                N_META,
                output_dir,
                steps,
                class_healthy_label,
                class_unhealthy_label,
                class_healthy_target,
                class_unhealthy_target,
                clf_name="SVM",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
            )
            process_data_frame_svm(
                output_dir,
                animal_ids,
                output_dir,
                df_processed,
                days,
                farm_id,
                step_slug,
                n_splits,
                n_repeats,
                sampling,
                enable_downsample_df,
                label_series,
                class_healthy_target,
                class_unhealthy_target,
                cv=cv,
            )
        # 2DCNN
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

        # 1DCNN
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

    output_dir = output_dir / cv
    files = [
        output_dir / file
        for file in os.listdir(output_dir)
        if file.endswith(".csv")
    ]
    print("found %d files." % len(files))
    print("compiling final file...")

    dfs = [pd.read_csv(file, sep=",") for file in files]
    df_final = pd.concat(dfs)
    filename = output_dir / "final_classification_report.csv"
    df_final.to_csv(filename, sep=",", index=False)
    print(df_final)
    plotMlReport(filename, output_dir)


if __name__ == "__main__":
    typer.run(main)
