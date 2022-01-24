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

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    plot_roc_curve,
    auc,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    LeaveOneOut,
    cross_validate,
)
from sklearn.svm import SVC

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import process_data_frame_svm

from utils._custom_split import StratifiedLeaveTwoOut, RepeatedKFoldCustom
from utils.visualisation import (
    plotMlReport,
)
from utils.visualisation import (
    plot_roc_range,
    plot_pr_range,
    plotHeatmap,
    plot_2D_decision_boundaries,
    plot_3D_decision_boundaries,
)

from preprocessing.preprocessing import apply_preprocessing_steps


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    preprocessing_steps: List[str] = [["MRNN", "QN", "ANSCOMBE", "LOG"]],
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    imputed_days: int = 7,
    n_scales: int = 8,
    hum_file: Optional[Path] = Path("."),
    temp_file: Optional[Path] = Path("."),
    n_splits: int = 2,
    n_repeats: int = 2,
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
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            N_META,
            class_healthy_target,
            class_unhealthy_target,
            label_series,
            samples,
        ) = load_activity_data(
            output_dir,
            file,
            day,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=imputed_days,
            preprocessing_steps=preprocessing_steps,
        )

        # plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
        ###############
        # VISUALISATION
        ###############
        # animal_ids = data_frame.iloc[0: len(data_frame), :]["id"].astype(str).tolist()
        # df_norm = apply_preprocessing_steps(
        #     days,
        #     df_hum,
        #     df_temp,
        #     sfft_window,
        #     wavelet_f0,
        #     animal_ids,
        #     data_frame.copy(),
        #     N_META,
        #     output_dir,
        #     ["QN"],
        #     class_healthy_label,
        #     class_unhealthy_label,
        #     class_healthy_target,
        #     class_unhealthy_target,
        #     clf_name="SVM_QN_VISU",
        #     n_scales=n_scales
        # )
        # #df_meta = data_frame.iloc[:, -N_META:]
        # #df_norm_meta = pd.concat([df_norm, df_meta], axis=1)
        #
        #
        # plot_zeros_distrib(
        #     label_series,
        #     df_norm,
        #     output_dir,
        #     title="Percentage of zeros in activity per sample after normalisation",
        # )
        # plot_zeros_distrib(
        #     label_series,
        #     data_frame.copy(),
        #     output_dir,
        #     title="Percentage of zeros in activity per sample before normalisation",
        # )
        # plotMeanGroups(
        #     n_scales,
        #     sfft_window,
        #     wavelet_f0,
        #     df_norm,
        #     label_series,
        #     N_META,
        #     output_dir / "raw_after_qn",
        # )
        #
        # plot_time_pca(
        #     N_META,
        #     data_frame.copy(),
        #     output_dir,
        #     label_series,
        #     title="PCA time domain before normalisation",
        # )
        # plot_time_pca(
        #     N_META,
        #     df_norm,
        #     output_dir,
        #     label_series,
        #     title="PCA time domain after normalisation",
        # )
        #
        # plot_time_lda(
        #     N_META,
        #     data_frame.copy(),
        #     output_dir,
        #     label_series,
        #     title="LDA time domain before normalisation",
        # )
        # plot_time_lda(
        #     N_META,
        #     data_frame.copy(),
        #     output_dir,
        #     label_series,
        #     title="LDA time domain after normalisation",
        # )
        #
        # ntraces = 2
        # idx_healthy, idx_unhealthy = plot_groups(
        #     N_META,
        #     animal_ids,
        #     class_healthy_label,
        #     class_unhealthy_label,
        #     class_healthy_target,
        #     class_unhealthy_target,
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
        #     class_healthy_target,
        #     class_unhealthy_target,
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
        # keep only two class of samples
        # data_frame = data_frame[
        #     data_frame["target"].isin([class_healthy_target, class_unhealthy_target])
        # ]
        animal_ids = data_frame.iloc[0 : len(data_frame), :]["id"].astype(str).tolist()
        # cv = "StratifiedLeaveTwoOut"

        preprocessing_steps = [""]
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

            model_files = process_data_frame_svm(
                N_META,
                output_dir,
                animal_ids,
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
                class_healthy_label,
                class_unhealthy_label,
                cv=cv,
            )

            for model_file in model_files:
                with open(str(model_file), "rb") as f:
                    clf = pickle.load(f)

                predict_list = []
                test_labels = []
                test_size = []
                for test_label, X in samples.items():
                    print(f"eval model {test_label} {model_file}...")
                    df_processed = apply_preprocessing_steps(
                        days,
                        df_hum,
                        df_temp,
                        sfft_window,
                        wavelet_f0,
                        animal_ids,
                        X.copy(),
                        N_META,
                        output_dir,
                        steps,
                        class_healthy_label,
                        class_unhealthy_label,
                        class_healthy_target,
                        class_unhealthy_target,
                        clf_name="SVM",
                        output_dim=data_frame.shape[0],
                        n_scales=None,
                        keep_meta=False,
                    )

                    X_test = df_processed.iloc[:, :-1].values
                    y_test = df_processed["target"].values
                    # predict = clf.predict(X_test)
                    predict = clf.predict_proba(X_test)[:, 1]
                    predict_list.append(predict)
                    test_labels.append(test_label)
                    test_size.append(X_test.shape[0])

                    print(predict)
                    plt.clf()
                    plt.xlabel("Probability to be unhealthy(2To2)", size=14)
                    plt.ylabel("Count", size=14)
                    plt.hist(predict, bins=30)
                    plt.title(
                        f"Histogram of predictions test_label={test_label} test_size={X_test.shape[0]}"
                    )

                    filename = f"{test_label}_{model_file.stem}.png"
                    out = output_dir / filename
                    print(out)
                    plt.savefig(str(out))

                i_h = 0
                i_uh = 0
                for bin_size in [5, 10, 30, 50, 100]:
                    plt.clf()
                    test_label_str = ""
                    test_size_str = ""
                    for i, (data, label, size) in enumerate(
                        zip(predict_list, test_labels, test_size)
                    ):
                        test_label_str += label + ","
                        test_size_str += str(size) + ","
                        if label in class_healthy_label:
                            i_h = i
                            continue
                        if label in class_unhealthy_label:
                            i_uh = i
                            continue
                        plt.hist(
                            data,
                            bins=bin_size,
                            alpha=0.5,
                            label=f"{label}({str(size)})",
                        )

                    plt.hist(
                        predict_list[i_h],
                        bins=bin_size,
                        alpha=0.5,
                        label=f"{test_labels[i_h]}({str(test_size[i_h])})",
                    )
                    plt.hist(
                        predict_list[i_uh],
                        bins=bin_size,
                        alpha=0.5,
                        label=f"{test_labels[i_uh]}({str(test_size[i_uh])})",
                    )

                    plt.xlabel("Probability to be unhealthy(2To2)", size=14)
                    plt.ylabel("Count", size=14)
                    plt.title(
                        f"Histogram of predictions (bin size={bin_size})\n test_label={test_label_str[:-1]} test_size={test_size_str[:-1]}"
                    )
                    plt.legend(loc="upper right")
                    filename = f"binsize_{bin_size}_{test_label_str.replace(',', '_')}_{model_file.stem}.png"
                    out = output_dir / filename
                    print(out)
                    plt.savefig(str(out))

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
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        output_dir / file for file in os.listdir(output_dir) if file.endswith(".csv")
    ]
    print("found %d files." % len(files))
    if len(files) == 0:
        return
    print("compiling final file...")

    dfs = [pd.read_csv(file, sep=",") for file in files]
    df_final = pd.concat(dfs)
    filename = output_dir / "final_classification_report.csv"
    df_final.to_csv(filename, sep=",", index=False)
    print(df_final)
    plotMlReport(filename, output_dir)

if __name__ == "__main__":
    # typer.run(main)

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

    for steps in [
        [["QN", "ANSCOMBE", "LOG"]],
        [["LINEAR"]],
        [["LINEAR", "QN"]],
        [["LINEAR", "QN", "ANSCOMBE"]],
        [["LINEAR", "QN", "ANSCOMBE", "LOG"]],
    ]:
        slug = "_".join(steps[0])
        main(
            output_dir=Path(f"E:\Data2\debugfinal3\delmas_{slug}"),
            dataset_folder=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
            preprocessing_steps=steps,
        )

    # for steps in [[['MRNN']], [['MRNN', "QN"]], [['MRNN', "QN", "ANSCOMBE"]],
    #               [['MRNN', "QN", "ANSCOMBE", "LOG"]]]:
    #     slug = '_'.join(steps[0])
    #     for day in [0, 1, 2, 3, 4, 5, 6, 7]:
    #         main(output_dir=Path(f"E:\Data2\debugfinal3\cedara_{day}_{slug}"),
    #              dataset_folder=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"), preprocessing_steps=steps,
    #              imputed_days=day, class_unhealthy_label=["2To4", "3To4", "1To4", "1To3", "4To5", "2To3"])
    #
    # for steps in [[['ZEROPAD']], [['ZEROPAD', "QN"]], [['ZEROPAD', "QN", "ANSCOMBE"]], [['ZEROPAD', "QN", "ANSCOMBE", "LOG"]],
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
