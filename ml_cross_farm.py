import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import makeRocCurve, processSVM
from preprocessing.preprocessing import applyPreprocessingSteps
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
    class_healthy: str = "1To1",
    class_unhealthy: str = "2To2",
    steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
):
    """This script train a ml model(SVM) on all the data of 1 dataset and test on a different dataset\n
    Args:\n
        farm1_path: Dataset input directory
        farm2_path: Dataset input directory
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
    """

    print(f"farm_1 {farm1_path}")
    print(f"farm_2{farm2_path}")

    days, farm_id, option, sampling = parse_param_from_filename(str(farm1_path))
    (
        dataset1,
        N_META,
        class_healthy_target,
        class_unhealthy_target,
        label_series,
    ) = load_activity_data(
        find_dataset(str(farm1_path)), days, class_healthy, class_unhealthy, keep_2_only=True
    )
    dataset2, _, _, _, _ = load_activity_data(
        find_dataset(str(farm2_path)), days, class_healthy, class_unhealthy, keep_2_only=True
    )

    print(dataset1)
    print(dataset2)

    dataframe = pd.concat([dataset1, dataset2], axis=0)
    #dataframe = dataframe["target"].isin([class_healthy_target, class_unhealthy_target])

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

    df1_processed = applyPreprocessingSteps(days, None, None, None, None, None,
                                           dataset1.copy(), N_META, output_dir, steps,
                                           class_healthy, class_unhealthy, class_healthy_target,
                                           class_unhealthy_target, clf_name="SVM", output_dim=dataset1.shape[0],
                                           n_scales=None, farm_name="FARM1")

    df2_processed = applyPreprocessingSteps(days, None, None, None, None, None,
                                           dataset2.copy(), N_META, output_dir, steps,
                                           class_healthy, class_unhealthy, class_healthy_target,
                                           class_unhealthy_target, clf_name="SVM", output_dim=dataset2.shape[0],
                                           n_scales=None, farm_name="FARM2")
    print(df1_processed)
    print(df2_processed)

    X1, y1 = getXY(df1_processed)
    X2, y2 = getXY(df2_processed)

    slug = "_".join(steps)
    clf_best, X, y = processSVM(X1, X2, y1, y2, output_dir)
    makeRocCurve(
        str(clf_best),
        output_dir,
        clf_best,
        X,
        y,
        None,
        slug,
        "Split",
        None,
        days,
        split1=y1.size,
        split2=y2.size,
    )


if __name__ == "__main__":
    typer.run(main)
