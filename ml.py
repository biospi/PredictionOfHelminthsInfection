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

from model.data_loader import load_activity_data
from model.svm import process_data_frame_svm
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import (
    plotHeatmap, plot_zeros_distrib, plot_mean_groups, plot_time_pca, plot_time_lda, plot_groups,
    plot_umap)


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    preprocessing_steps: List[str] = [["QN", "ANSCOMBE", "LOG"]],
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    meta_columns: List[str] = ["label", "id", "imputed_days", "date", "health", "target"],
    n_imputed_days: int = 7,
    n_activity_days: int = 7,
    n_scales: int = 9,
    hum_file: Optional[Path] = Path("."),
    temp_file: Optional[Path] = Path("."),
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedStratifiedKFold",
    wavelet_f0: int = 6,
    sfft_window: int = 60,
    farm_id: str = "farm",
    sampling: str = "T",
    pre_visu: bool = True,
    n_job: int = 6,
):
    """ML Main machine learning script\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
        meta_columns: List of names of the metadata columns in the dataset file.
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
        n_job: Number of threads to use for cross validation.
    """
    enable_downsample_df = False

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
        #_, _, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            meta_data,
            _,
            _,
            label_series,
            samples,
        ) = load_activity_data(
            output_dir,
            meta_columns,
            file,
            n_activity_days,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=n_imputed_days,
            preprocessing_steps=preprocessing_steps,
        )

        if pre_visu:
            # plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
            #############################################################################################################
            #                                           VISUALISATION                                                   #
            #############################################################################################################
            animal_ids = data_frame.iloc[0: len(data_frame), :]["id"].astype(str).tolist()
            N_META = len(meta_columns)

            df_norm = apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
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
                output_qn_graph=True
            )

            plot_zeros_distrib(
                n_activity_days,
                label_series,
                df_norm,
                output_dir,
                title="Percentage of zeros in activity per sample after normalisation",
            )
            plot_zeros_distrib(
                n_activity_days,
                label_series,
                data_frame.copy(),
                output_dir,
                title="Percentage of zeros in activity per sample before normalisation",
            )
            plot_mean_groups(
                n_scales,
                sfft_window,
                wavelet_f0,
                df_norm,
                label_series,
                N_META,
                output_dir / "groups_after_qn",
            )

            #plot median wise cwt for each target(label)
            apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
                wavelet_f0,
                animal_ids,
                df_norm.copy(),
                output_dir / "groups_after_qn" ,
                ["ANSCOMBE", "LOG", "CENTER", "CWT"],
                class_healthy_label,
                class_unhealthy_label,
                clf_name="QN_CWT_VISU",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
                keep_meta=True,
                plot_all_target=True,
                enable_graph_out=False
            )
            apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
                wavelet_f0,
                animal_ids,
                df_norm.copy(),
                output_dir / "groups_after_qn" ,
                ["ANSCOMBE", "LOG", "CENTER", "CWT", "LOG"],
                class_healthy_label,
                class_unhealthy_label,
                clf_name="QN_CWT_LOG_VISU",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
                keep_meta=True,
                plot_all_target=True,
                enable_graph_out=False
            )

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

        sample_dates = pd.to_datetime(data_frame['date'], format="%d/%m/%Y").values.astype(float)
        animal_ids = data_frame.iloc[0: len(data_frame), :]["id"].astype(str).tolist()

        for steps in preprocessing_steps:
            step_slug = "_".join(steps)
            df_processed = apply_preprocessing_steps(
                meta_columns,
                n_activity_days,
                df_hum,
                df_temp,
                sfft_window,
                wavelet_f0,
                animal_ids,
                data_frame.copy(),
                output_dir,
                steps,
                class_healthy_label,
                class_unhealthy_label,
                clf_name="SVM",
                output_dim=data_frame.shape[0],
                n_scales=n_scales,
            )

            process_data_frame_svm(
                meta_data,
                output_dir,
                animal_ids,
                sample_dates,
                df_processed,
                n_activity_days,
                n_imputed_days,
                farm_id,
                step_slug,
                n_splits,
                n_repeats,
                sampling,
                enable_downsample_df,
                label_series,
                class_healthy_label,
                class_unhealthy_label,
                meta_columns,
                cv=cv,
                n_job=n_job
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
    #typer.run(main)

    # --output-dir E:\Data2\debug\mrnn_median_ml\debug --dataset-folder E:\Data2\debug\delmas\dataset_mrnn_7day
    # --output-dir E:\Data2\debug\mrnn_median_ml\debug\cedara --dataset-folder E:\Data2\debug\delmas\dataset_mrnn_7day
    # --imputed-days 4 --output-dir E:\Data2\mrnn_median_ml\10080_100_2To2_4 --dataset-folder F:\Data2\mrnn_datasets2\median_10080_100\dataset_gain_7day

    # for i in [7, 6, 5, 4, 3, 2, 1, 0]:
    #     for healthy_l, unhealthy_l in zip([['1To1', '1To2'], ['1To1', '2To1'], ['1To1', '2To1'], ['1To1'], ['1To1']], [['2To2'], ['2To2'], ['2To2', '1To2'], ['2To2'], ['1To2']]):
    #         main(imputed_days=i, output_dir=Path(f"E:/Data2/mrnn_median_ml_weather/10080_100_{healthy_l}__{unhealthy_l}_{i}"),
    #              dataset_folder=Path("E:\Data2\mrnn_datasets2\median_10080_100_weather\dataset_gain_7day"),
    #              class_healthy_label=healthy_l, class_unhealthy_label=unhealthy_l)

    # for steps in [[["ZEROPAD", "QN"]], [["ZEROPAD", "QN", "ANSCOMBE"]],
    #               [["ZEROPAD", "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     for day in [7]:
    #         main(output_dir=Path(f"E:\Data2\debugfinal3\delmas_li_{day}_{slug}"),
    #              dataset_folder=Path("E:\Data2\debug3\delmas\dataset_li_7day"), preprocessing_steps=steps, imputed_days=day)

    # for steps in [[['MRNN']], [['MRNN', "QN"]], [['MRNN', "QN", "ANSCOMBE"]],
    #               [['MRNN', "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     for day in [0, 1, 2, 3, 4, 5, 6, 7]:
    #         main(output_dir=Path(f"E:\Data2\debugfinal3\delmas_{day}_{slug}"),
    #              dataset_folder=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"), preprocessing_steps=steps, imputed_days=day)

    # for steps in [[['GAIN']], [['GAIN', "QN"]], [['GAIN', "QN", "ANSCOMBE"]],
    #               [['GAIN', "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     main(output_dir=Path(f"E:\Data2\debugfinal3\delmas_{slug}"),
    #          dataset_folder=Path("E:\Data2\debug3\delmas\dataset_gain_7day"), preprocessing_steps=steps)

    # for steps in [
    #     [["QN", "ANSCOMBE", "LOG"]],
    #     [["LINEAR"]],
    #     [["LINEAR", "QN"]],
    #     [["LINEAR", "QN", "ANSCOMBE"]],
    #     [["LINEAR", "QN", "ANSCOMBE", "LOG"]],
    # ]:
    #     slug = "_".join(steps[0])
    #     main(
    #         output_dir=Path(f"E:\Data2\debugfinal3\delmas_{slug}"),
    #         dataset_folder=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
    #         preprocessing_steps=steps,
    #     )
    #
    # for steps in [[["QN", "ANSCOMBE", "LOG"]],
    #               [['MRNN']], [['MRNN', "QN"]], [['MRNN', "QN", "ANSCOMBE"]],
    #               [['MRNN', "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     for day in [7]:
    #         main(output_dir=Path(f"E:\Data2\debugfinal3\cedara_{day}_{slug}"),
    #              dataset_folder=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"), preprocessing_steps=steps,
    #              imputed_days=day, class_unhealthy_label=["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"])

    # for steps in [[["QN", "ANSCOMBE", "LOG"]],
    #               [['ZEROPAD']], [['ZEROPAD', "QN"]], [['ZEROPAD', "QN", "ANSCOMBE"]], [['ZEROPAD', "QN", "ANSCOMBE", "LOG"]],
    #               [['LINEAR']], [['LINEAR', "QN"]], [['LINEAR', "QN", "ANSCOMBE"]], [['LINEAR', "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     main(output_dir=Path(f"E:\Data2\debugfinal3\cedara_{slug}"),
    #          dataset_folder=Path("E:\Data2\debug3\cedara_\datasetraw_none_7day"), preprocessing_steps=steps,
    #          class_unhealthy_label=["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"])

    # output_dir = Path(f"E:\Data2\debugfinal3")
    # files = output_dir.rglob("*.csv")
    # dfs = []
    # for file in files:
    #     if 'final_classification_report' in str(file):
    #         continue
    #     print(file)
    #     dfs.append(pd.read_csv(file))
    # #dfs = [pd.read_csv(file, sep=",") for file in files if 'final' not in file]
    # df_final = pd.concat(dfs).drop_duplicates()
    # filename = output_dir / "final_classification_report.csv"
    # df_final.to_csv(filename, sep=",", index=False)
    # print(df_final)
    # plotMlReport(filename, output_dir)

    steps = [["LINEAR", "QN", "ANSCOMBE", "LOG", "CENTER", "CWT"]]
    slug = "_".join(steps[0])
    day = 7
    main(
        output_dir=Path(f"E:\Data2\debugfinal3\delmas_{slug}"),
        dataset_folder=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
        preprocessing_steps=steps,
        n_imputed_days=0
    )
    #
    # main(output_dir=Path(f"E:\Data2\debugfinal3\cedara_{day}_{slug}"),
    #      dataset_folder=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"), preprocessing_steps=steps,
    #      imputed_days=day, class_unhealthy_label=["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"])
    # main(output_dir=Path(f"E:/Cats/ml/ml_min/day_w"),
    #      dataset_folder=Path("E:/Cats/build_min/dataset/training_sets/day_w"), preprocessing_steps=steps,
    #      n_imputed_days=-1, n_activity_days=None, class_healthy_label=['0'], class_unhealthy_label=['1'], n_splits=5, n_repeats=2, n_job=5)