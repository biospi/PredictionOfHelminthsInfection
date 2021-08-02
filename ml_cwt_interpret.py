import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from classifier.src.cwt_weight import (
    reduce_lda,
    chunck_df,
    explain_cwt,
    get_cwt_data_frame,
)
from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import applyPreprocessingSteps
from utils.Utils import getXY


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    class_healthy_label: str = "1To1",
    class_unhealthy_label: str = "2To2",
    steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
):
    """This script builds the graphs for cwt interpretation\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
    """
    files = glob.glob(str(dataset_folder / "*.csv"))  # find datset files
    print("found %d files." % len(files))
    print(files)

    for file in files:
        days, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            N_META,
            class_healthy_target,
            class_unhealthy_target,
            label_series,
        ) = load_activity_data(file, days, class_healthy_label, class_unhealthy_label)

        print(data_frame)

        data_frame = applyPreprocessingSteps(
            days,
            None,
            None,
            None,
            None,
            None,
            data_frame.copy(),
            N_META,
            output_dir,
            steps,
            class_healthy_label,
            class_unhealthy_label,
            class_healthy_target,
            class_unhealthy_target,
            clf_name="SVM",
            n_scales=10,
            farm_name="FARMS",
            keep_meta=False,
        )
        print(data_frame)
        X, y = getXY(data_frame)

        df_cwt, cwt_coefs_data = get_cwt_data_frame(data_frame)
        dfs, data = chunck_df(days, df_cwt, cwt_coefs_data, ignore=True, W_DAY_STEP=1)

        explain_cwt(
            dfs,
            data,
            None,
            None,
            "cwt.png",
            df_temp=None,
            df_hum=None,
            resolution=None,
            farm_id=farm_id,
            days=days,
            f_config=None,
            out_dir=output_dir,
        )


if __name__ == "__main__":
    typer.run(main)
