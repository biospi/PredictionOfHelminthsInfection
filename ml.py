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

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sys import exit
from model.data_loader import load_activity_data
from model.svm import process_ml
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import (
    plotHeatmap,
    plot_zeros_distrib,
    plot_mean_groups,
    plot_time_pca,
    plot_time_lda,
    plot_groups,
    plot_umap,
    plot_time_pls, plot_ml_report_final)


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
    ],
    meta_col_str: List[str] = ["health", "label", "id"],
    add_feature: List[str] = [],
    n_imputed_days: int = 7,
    n_activity_days: int = 7,
    n_scales: int = 9,
    sub_sample_scales: int = 3,
    hum_file: Optional[Path] = Path("."),
    temp_file: Optional[Path] = Path("."),
    add_seasons_to_features: bool = False,
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedStratifiedKFold",
    classifiers: List[str] = [],
    wavelet_f0: int = 6,
    sfft_window: int = 60,
    dwt_window: str = "coif1",
    study_id: str = "study",
    sampling: str = "T",
    pre_visu: bool = True,
    output_qn_graph: bool = False,
    enable_downsample_df: bool = False,
    n_job: int = 7,
    batch_size: int = 100,
    epoch: int = 100,
    individual_to_ignore: List[str] = [],
    individual_to_keep: List[str] = [],
    individual_to_test: List[str] = [],
    save_model: bool = False,
    resolution: float = None
):
    """ML Main machine learning script\n
    Args:\n
        output_dir: Output directory.
        dataset_folder: Dataset input directory.
        preprocessing_steps: preprocessing steps.
        class_healthy_label: Label for healthy class.
        class_unhealthy_label: Label for unhealthy class.
        meta_columns: List of names of the metadata columns in the dataset file.plo
        meta_col_str: List of meta data to display on decision boundary plot.
        n_imputed_days: number of imputed days allowed in a sample.
        n_activity_days: number of activity days in a sample.
        n_scales: n scales in dyadic array [2^2....2^n].
        temp_file: csv file containing temperature features.
        hum_file: csv file containing humidity features.
        n_splits: Number of splits for repeatedkfold cv.
        n_repeats: Number of repeats for repeatedkfold cv.
        cv: RepeatedKFold|LeaveOneOut.
        wavelet_f0: Mother Wavelet frequency for CWT.
        sfft_window: STFT window size.
        farm_id: farm id.
        sampling: Activity bin resolution.
        pre_visu: Enable initial visualisation of dataset before classification.
        output_qn_graph: Output Quotient Normalisation steps figures.
        n_job: Number of threads to use for cross validation.
    """

    meta_columns = [x.replace("'", '') for x in meta_columns]
    preprocessing_steps = [x.replace("'", '') for x in preprocessing_steps]
    print(f"meta_columns={meta_columns}")
    print(f"preprocessing_steps={preprocessing_steps}")
    print(f"output directory is {output_dir}")
    # if output_dir.is_dir():
    #     print("output directory already exist. skip run.")
    #     return

    files = [str(x) for x in list(dataset_folder.glob("*.csv"))]  # find datset files

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
        # _, _, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (data_frame, meta_data, meta_data_short, _, _, label_series, samples, seasons_features) = load_activity_data(
            output_dir,
            meta_columns,
            file,
            n_activity_days,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=n_imputed_days,
            preprocessing_steps=preprocessing_steps,
            meta_cols_str=meta_col_str,
            sampling=sampling,
            individual_to_keep=individual_to_keep,
            resolution=resolution
        )

        N_META = len(meta_columns)
        if pre_visu:
            # plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
            #############################################################################################################
            #                                           VISUALISATION                                                   #
            #############################################################################################################
            animal_ids = data_frame["id"].astype(str).tolist()
            df_norm, _, time_freq_shape = apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
                dwt_window,
                wavelet_f0,
                animal_ids,
                data_frame.copy(),
                output_dir,
                ["QN"],
                class_healthy_label,
                class_unhealthy_label,
                clf_name="SVM_QN_VISU",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
                keep_meta=True,
                output_qn_graph=output_qn_graph,
                sub_sample_scales=sub_sample_scales
            )

            # plot_zeros_distrib(
            #     meta_columns,
            #     n_activity_days,
            #     label_series,
            #     df_norm,
            #     output_dir,
            #     title="Percentage of zeros in activity per sample after normalisation",
            # )
            # plot_zeros_distrib(
            #     meta_columns,
            #     n_activity_days,
            #     label_series,
            #     data_frame.copy(),
            #     output_dir,
            #     title="Percentage of zeros in activity per sample before normalisation",
            # )
            plot_mean_groups(
                sub_sample_scales,
                n_scales,
                sfft_window,
                wavelet_f0,
                dwt_window,
                df_norm,
                label_series,
                N_META,
                output_dir / "groups_after_qn",
            )

            # plot median wise cwt for each target(label)
            # apply_preprocessing_steps(
            #     meta_columns,
            #     n_activity_days,
            #     df_hum,
            #     df_temp,
            #     sfft_window,
            #     wavelet_f0,
            #     animal_ids,
            #     df_norm.copy(),
            #     output_dir / "groups_after_qn",
            #     ["ANSCOMBE", "LOG", "CENTER", "CWT"],
            #     class_healthy_label,
            #     class_unhealthy_label,
            #     clf_name="QN_CWT_VISU",
            #     output_dim=data_frame.shape[0],
            #     n_scales=n_scales,
            #     keep_meta=True,
            #     plot_all_target=True,
            #     enable_graph_out=False,
            #     sub_sample_scales=sub_sample_scales
            # )

            plot_umap(
                meta_columns,
                data_frame.copy(),
                output_dir / "umap",
                label_series,
                title="UMAP time domain before normalisation",
            )

            plot_umap(
                meta_columns,
                df_norm.copy(),
                output_dir / "umap",
                label_series,
                title="UMAP time domain after normalisation",
            )

            plot_time_pca(
                N_META,
                data_frame.copy(),
                output_dir / "pca",
                label_series,
                title="PCA time domain before normalisation",
            )
            plot_time_pca(
                N_META,
                df_norm,
                output_dir / "pca",
                label_series,
                title="PCA time domain after normalisation",
            )

            plot_time_pls(
                N_META,
                data_frame.copy(),
                output_dir / "pls",
                label_series,
                title="PLS time domain before normalisation",
            )
            plot_time_pls(
                N_META,
                df_norm,
                output_dir / "pls",
                label_series,
                title="PLS time domain after normalisation",
            )

            plot_time_lda(
                N_META,
                data_frame.copy(),
                output_dir / "lda",
                label_series,
                title="LDA time domain before normalisation",
            )
            plot_time_lda(
                N_META,
                data_frame.copy(),
                output_dir / "lda",
                label_series,
                title="LDA time domain after normalisation",
            )

            # ntraces = 2
            # idx_healthy, idx_unhealthy = plot_groups(
            #     N_META,
            #     animal_ids,
            #     class_healthy_label,
            #     class_unhealthy_label,
            #     output_dir,
            #     data_frame.copy(),
            #     title="Raw imputed",
            #     xlabel="Time",
            #     ylabel="activity",
            #     ntraces=ntraces,
            # )
            # plot_groups(
            #     N_META,
            #     animal_ids,
            #     class_healthy_label,
            #     class_unhealthy_label,
            #     output_dir,
            #     df_norm,
            #     title="Normalised(Quotient Norm) samples",
            #     xlabel="Time",
            #     ylabel="activity",
            #     idx_healthy=idx_healthy,
            #     idx_unhealthy=idx_unhealthy,
            #     stepid=2,
            #     ntraces=ntraces,
            # )
            ################################################################################################################

        sample_dates = pd.to_datetime(
            data_frame["date"], format="%d/%m/%Y"
        ).values.astype(float)
        animal_ids = data_frame["id"].astype(str).tolist()

        step_slug = "_".join(preprocessing_steps)

        steps_ = []
        if 'APPEND' in preprocessing_steps:
            steps_.append(preprocessing_steps[0:preprocessing_steps.index('APPEND')])
            steps_.append(preprocessing_steps[preprocessing_steps.index('APPEND') + 1:])
        else:
            steps_ = [preprocessing_steps]

        df_processed_list = []
        for preprocessing_steps in steps_:
            df_processed, df_processed_meta, time_freq_shape = apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
                dwt_window,
                wavelet_f0,
                animal_ids,
                data_frame.copy(),
                output_dir,
                preprocessing_steps,
                class_healthy_label,
                class_unhealthy_label,
                clf_name="SVM",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
                sub_sample_scales=sub_sample_scales
            )
            df_processed_list.append(df_processed)

        df = pd.concat(df_processed_list, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        target = df.pop('target')
        health = df.pop('health')
        df_processed = pd.concat([df, target, health], 1)

        # plot_umap(
        #     meta_columns,
        #     df_processed_meta.copy(),
        #     output_dir / f"umap_{step_slug}",
        #     label_series,
        #     title=f"UMAP after {step_slug}",
        # )
        #
        # plot_time_pca(
        #     N_META,
        #     df_processed_meta,
        #     output_dir / f"pca_{step_slug}",
        #     label_series,
        #     title=f"PCA after {step_slug}",
        # )
        #
        # plot_time_pls(
        #     N_META,
        #     df_processed_meta.copy(),
        #     output_dir / f"pls_{step_slug}",
        #     label_series,
        #     title=f"PLS after {step_slug}",
        # )

        if len(add_feature) > 0:
            df_meta = pd.DataFrame(meta_data, columns=meta_columns, index=df_processed.index)[add_feature]
            df_data = df_processed[df_processed.columns[~df_processed.columns.isin(["target", "health"])]]
            df_ = pd.concat([df_data, df_meta], axis=1)
            #df_ = pd.concat([df_meta], axis=1)
            #df_ = pd.DataFrame(StandardScaler().fit_transform(df_))
            df_processed = pd.concat([df_, df_processed[["target", "health"]]], axis=1)
            step_slug = f"{step_slug}_{'_'.join(add_feature).upper()}_STDS"

        if add_seasons_to_features:
            df_data = df_processed[df_processed.columns[~df_processed.columns.isin(["target", "health"])]]
            df_target = df_processed[["target", "health"]]
            df_processed = pd.concat([df_data, seasons_features, df_target], axis=1)
            step_slug = f"{step_slug}_SEASONS"

        process_ml(
            classifiers,
            add_feature,
            meta_data,
            meta_data_short,
            output_dir,
            animal_ids,
            sample_dates,
            df_processed,
            n_activity_days,
            n_imputed_days,
            study_id,
            step_slug,
            n_splits,
            n_repeats,
            sampling,
            enable_downsample_df,
            label_series,
            class_healthy_label,
            class_unhealthy_label,
            meta_columns,
            add_seasons_to_features,
            save_model=save_model,
            cv=cv,
            n_job=n_job,
            batch_size=batch_size,
            epoch=epoch,
            time_freq_shape=time_freq_shape,
            individual_to_test=individual_to_test
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


if __name__ == "__main__":
    typer.run(main)




