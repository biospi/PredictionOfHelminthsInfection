import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import loadActivityData, parse_param_from_filename
from model.svm import makeRocCurve, processSVM
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
    steps: List[str] = ["QN", "ANSCOMBE", "LOG", "DIFF"],
):
    """This script train a ml model(SVM) on the dataset first half time period and test on the second half\n
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
        ) = loadActivityData(file, days, class_healthy_label, class_unhealthy_label)

        data_frame = data_frame[
            data_frame["target"].isin([class_healthy_target, class_unhealthy_target])
        ]

        data_frame["date_"] = pd.to_datetime(data_frame["date"], dayfirst=True)
        data_frame = data_frame.sort_values("date_", ascending=True)
        del data_frame["date_"]

        # print(data_frame)
        nrows = int(data_frame.shape[0] / 2)
        print(nrows)
        print(
            "data_frame:%s %s"
            % (
                str(data_frame["date"].iloc[0]).split(" ")[0],
                str(data_frame["date"].iloc[-1]).split(" ")[0],
            )
        )

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
            n_scales=None,
            farm_name="FARMS",
            keep_meta=False,
        )

        df1 = data_frame.iloc[:nrows, :]
        df2 = data_frame.iloc[nrows:, :]

        print(df1)
        print(df2)
        X1, y1 = getXY(df1)
        X2, y2 = getXY(df2)
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
