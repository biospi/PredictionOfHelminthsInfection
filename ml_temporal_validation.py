import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import make_roc_curve, processSVM
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.Utils import getXY, plot_heatmap


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    imputed_days: int = 5,
    class_healthy_label: str = "1To1",
    class_unhealthy_label: str = "2To2",
    steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
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
            samples,
        ) = load_activity_data(file, days, class_healthy_label, class_unhealthy_label, imputed_days=imputed_days)

        data_frame = data_frame[
            data_frame["target"].isin([class_healthy_target, class_unhealthy_target])
        ]

        data_frame["date_"] = pd.to_datetime(data_frame["date"], dayfirst=True)
        data_frame = data_frame.sort_values("date_", ascending=True)
        del data_frame["date_"]

        # print(data_frame)
        nrows = int(data_frame.shape[0] / 2)
        print(nrows)
        p1_start = str(data_frame["date"].iloc[0]).split(" ")[0]
        p1_end = str(data_frame["date"].iloc[nrows]).split(" ")[0]
        p2_start = p1_end
        p2_end = str(data_frame["date"].iloc[-1]).split(" ")[0]

        print(
            "data_frame:%s %s %s"
            % (
                p1_start,
                p1_end,
                p2_end
            )
        )

        data_frame = apply_preprocessing_steps(
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


        plot_heatmap(X1, y1, X2, y2, output_dir, p1_start, p1_end, p2_start, p2_end)

        slug = "_".join(steps)

        clf_best, X, y = processSVM(X1, X2, y1, y2, output_dir)
        make_roc_curve(
            str(clf_best),
            output_dir,
            clf_best,
            X,
            y,
            None,
            slug,
            "split",
            None,
            days
        )

        clf_best, X, y = processSVM(X2, X1, y2, y1, output_dir)
        make_roc_curve(
            str(clf_best),
            output_dir,
            clf_best,
            X,
            y,
            None,
            slug,
            "split",
            None,
            days,
            tag='rev'
        )


if __name__ == "__main__":
    #typer.run(main)

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        main(Path(f'E:/Data2/debug/temporal/{i}'), Path('E:/Data2/debug/delmas/dataset_mrnn_7day'), i)
