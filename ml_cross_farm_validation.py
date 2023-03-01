import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import process_clf, process_clf_
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.Utils import getXY


def find_dataset(folder):
    files = glob.glob(folder + "/*.csv")  # find datset files
    files = [file.replace("\\", "/") for file in files]
    print("found %d files." % len(files))
    print(files)
    return files[0]


def main(
    farm1_path: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    farm2_path: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    class_healthy_f1: List[str] = ["1To1"],
    class_unhealthy_f1: List[str] = ["2To2"],
    class_healthy_f2: List[str] = ["1To1"],
    class_unhealthy_f2: List[str] = ["2To2"],
    steps: List[str] = ["QN", "ANSCOMBE", "LOG", "STDS"],
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
    ],
    meta_col_str: List[str] = ["health", "label", "date"],
    n_fold: int = 50,
    n_imputed_days: int = 7,
    n_activity_days: int = 7,
    train_size: float = 0.9,
    n_job: int = 1,
):
    """This script train a ml model(SVM) on all the data of 1 dataset and test on a different dataset\n
    Args:\n
        farm1_path: Dataset input directory
        farm2_path: Dataset input directory
        output_dir: Output directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
    """

    print(f"farm_1 {farm1_path}")
    print(f"farm_2{farm2_path}")
    info = {"farm1": {"healthy": class_healthy_f1, "unhealthy": class_unhealthy_f1},
            "farm2": {"healthy": class_healthy_f2, "unhealthy": class_unhealthy_f2}}
    print(info)

    _, farm_id, option, sampling = parse_param_from_filename(str(farm1_path))
    (
        dataset1,
        meta_data,
        meta_data_short,
        _,
        _,
        label_series_f1,
        samples_f1,
        _
    ) = load_activity_data(
        output_dir,
        meta_columns,
        find_dataset(str(farm1_path)),
        n_activity_days,
        class_healthy_f1,
        class_unhealthy_f1,
        imputed_days=n_imputed_days,
        meta_cols_str=meta_col_str,
        preprocessing_steps=steps
    )

    dataset2, _, _, _, _, label_series_f2, samples_f2, _ = load_activity_data(
        output_dir,
        meta_columns,
        find_dataset(str(farm2_path)),
        n_activity_days,
        class_healthy_f2,
        class_unhealthy_f2,
        meta_cols_str=meta_col_str,
        imputed_days=n_imputed_days,
        farm='cedara',
        preprocessing_steps=steps,
    )

    print(dataset1)
    print(dataset2)

    dataframe = pd.concat([dataset1, dataset2], axis=0)

    dfs_processed, _ , _ = apply_preprocessing_steps(
        meta_columns,
        n_activity_days,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        dataframe.copy(),
        output_dir,
        steps,
        class_healthy_f1,
        class_unhealthy_f1,
        clf_name="SVM",
        output_dim=dataset1.shape[0],
        n_scales=None,
        farm_name="FARM1+FARM2",

    )

    # dataframe = dataframe["target"].isin([class_healthy_target, class_unhealthy_target])

    # df_processed = applyPreprocessingSteps(
    #     days,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     dataframe.copy(),
    #     N_META,
    #     output_dir,
    #     steps,
    #     class_healthy,
    #     class_unhealthy,
    #     class_healthy_target,
    #     class_unhealthy_target,
    #     clf_name="SVM",
    #     output_dim=dataset1.shape[0],
    #     n_scales=None,
    #     farm_name="FARMS",
    # )
    #
    # df1_processed = df_processed.iloc[0: dataset1.shape[0], :]
    # df2_processed = df_processed.iloc[dataset1.shape[0]:, :]

    # df1_processed = apply_preprocessing_steps(
    #     days,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     dataset1.copy(),
    #     N_META,
    #     output_dir,
    #     steps,
    #     class_healthy_f1,
    #     class_unhealthy_f1,
    #     class_healthy_target,
    #     class_unhealthy_target,
    #     clf_name="SVM",
    #     output_dim=dataset1.shape[0],
    #     n_scales=None,
    #     farm_name="FARM1",
    # )
    #
    # df2_processed = apply_preprocessing_steps(
    #     days,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     dataset2.copy(),
    #     N_META,
    #     output_dir,
    #     steps,
    #     class_healthy_f2,
    #     class_unhealthy_f2,
    #     class_healthy_target,
    #     class_unhealthy_target,
    #     clf_name="SVM",
    #     output_dim=dataset2.shape[0],
    #     n_scales=None,
    #     farm_name="FARM2",
    # )

    df1_processed = dfs_processed.iloc[0:dataset1.shape[0], :]
    df2_processed = dfs_processed.iloc[dataset1.shape[0]:, :]

    print(df1_processed)
    print(df2_processed)

    X1, y1 = getXY(df1_processed)
    X2, y2 = getXY(df2_processed)

    process_clf(n_activity_days, train_size, label_series_f1, label_series_f2, info, steps, n_fold, X1, X2, y1, y2, output_dir, plot_2d_space=True)

    process_clf(n_activity_days, train_size, label_series_f2, label_series_f1, info, steps, n_fold, X2, X1, y2, y1, output_dir / 'rev', plot_2d_space=True)

    # for clf_best, X, y in results:
    #     make_roc_curve(
    #         class_healthy_target,
    #         class_unhealthy_target,
    #         str(clf_best),
    #         output_dir,
    #         clf_best,
    #         X,
    #         y,
    #         None,
    #         slug,
    #         "Split",
    #         None,
    #         days,
    #         split1=y1.size,
    #         split2=y2.size,
    #     )


if __name__ == "__main__":
    # typer.run(main)
    for imp_d in [7, 6, 5, 4, 3, 2, 1, 0]:
        for a_act_day in [7, 6, 5, 4, 3, 2, 1]:
            main(
                farm1_path=Path("E:\Data2\debug3\delmas\dataset4_mrnn_7day"),
                farm2_path=Path("E:\Data2\debug3\cedara\dataset6_mrnn_7day"),
                output_dir=Path(f"E:\Data2\debug3\cross_farm_{imp_d}_{a_act_day}"),
                n_imputed_days=imp_d,
                n_activity_days=a_act_day
            )
