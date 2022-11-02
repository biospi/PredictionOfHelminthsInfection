import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import process_clf, process_clf_
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.Utils import getXY, plot_heatmap
import numpy as np
from sys import exit


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    model_path: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    n_imputed_days: int = 6,
    n_activity_days: int = 7,
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
    meta_col_str: List[str] = ["health", "label", "date"],
    add_feature: List[str] = [],
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    train_size: float = 0.9,
    n_fold: int = 50,
    sample_date_filter: str = None,
    export_fig_as_pdf:bool = False
):
    """This script train a ml model(SVM) on the dataset first half time period and test on the second half\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
    """

    info = {"healthy": class_healthy_label, "unhealthy": class_unhealthy_label}
    print(info)

    files = glob.glob(str(dataset_folder / "*.csv"))  # find datset files
    print("found %d files." % len(files))
    print(files)

    for file in files:
        _, farm_id, option, sampling = parse_param_from_filename(file)
        print(f"loading dataset file {file} ...")
        (
            data_frame,
            meta_data,
            meta_data_short,
            _,
            _,
            label_series,
            samples,
            _
        ) = load_activity_data(
            output_dir,
            meta_columns,
            file,
            n_activity_days,
            class_healthy_label,
            class_unhealthy_label,
            imputed_days=n_imputed_days,
            preprocessing_steps=preprocessing_steps,
            meta_cols_str=meta_col_str,
            sample_date_filter=sample_date_filter
        )

        data_frame = data_frame[
            data_frame["health"].isin([0, 1])
        ]

        data_frame["date_"] = pd.to_datetime(data_frame["date"], dayfirst=True)
        data_frame = data_frame.sort_values("date_", ascending=True)
        del data_frame["date_"]

        # if 'cedara' in str(dataset_folder):
        #     data_frame = data_frame.iloc[23:, :]

        # print(data_frame)
        nrows = int(data_frame.shape[0] / 2)
        print(nrows)
        p1_start = str(data_frame["date"].iloc[0]).split(" ")[0]
        p1_end = str(data_frame["date"].iloc[nrows]).split(" ")[0]
        p2_start = p1_end
        p2_end = str(data_frame["date"].iloc[-1]).split(" ")[0]

        p1_start = pd.to_datetime(p1_start, format='%d/%m/%Y').strftime('%B %Y')
        p1_end = pd.to_datetime(p1_end, format='%d/%m/%Y').strftime('%B %Y')
        p2_start = pd.to_datetime(p2_start, format='%d/%m/%Y').strftime('%B %Y')
        p2_end = pd.to_datetime(p2_end, format='%d/%m/%Y').strftime('%B %Y')

        print(
            "data_frame: %s->%s->%s"
            % (
                p1_start,
                p1_end,
                p2_end
            )
        )

        data_frame, df_with_meta, _ = apply_preprocessing_steps(
            meta_columns,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            data_frame.copy(),
            output_dir,
            preprocessing_steps,
            class_healthy_label,
            class_unhealthy_label,
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

        unique, counts = np.unique(y1, return_counts=True)
        y1_stat = dict(zip(unique, counts))
        print(y1_stat)

        unique, counts = np.unique(y2, return_counts=True)
        y2_stat = dict(zip(unique, counts))
        print(y2_stat)

        plot_heatmap(X1, y1, X2, y2, output_dir, p1_start, p1_end, p2_start, p2_end)

        # process_clf_(preprocessing_steps, X1, y1, model_path, output_dir / "pre_trained" / f"{p1_start}{p1_end}_{p2_start}{p2_end}".replace("/", ""))
        # process_clf_(preprocessing_steps, X2, y2, model_path, output_dir / "pre_trained" / f"{p2_start}{p2_end}_{p1_start}{p1_end}".replace("/", ""))

        process_clf(n_activity_days, train_size, label_series, label_series, info, preprocessing_steps, n_fold, X1, X2, y1, y2, output_dir / f"{p1_start}_{p2_start}".replace("/", ""))
        process_clf(n_activity_days, train_size, label_series, label_series, info, preprocessing_steps, n_fold, X2, X1, y2, y1, output_dir / f"{p2_start}_{p1_start}".replace("/", ""))


if __name__ == "__main__":
    #typer.run(main)

    for i in [1, 2, 3, 4, 5, 6, 7]:
        # main(Path(f'E:/Data2/debug4/temporal/delmas/{i}'), Path("E:/Data2/debug3/delmas/dataset4_mrnn_7day"),
        #      Path('E:thesis/main_experiment/delmas_RepeatedKFold_7_7_QN_ANSCOMBE_LOG_season_False/2To2/models/SVC_linear_7_QN_ANSCOMBE_LOG'), n_imputed_days=i)

        main(Path(f'E:/Data2/debug4/temporal/cedara/{i}'), Path('E:/Data2/debug3/cedara/dataset6_mrnn_7day'),
             Path('E:/thesis/main_experiment/cedara_RepeatedKFold_7_7_QN_ANSCOMBE_LOG_season_False/2To2/models/SVC_linear_7_QN_ANSCOMBE_LOG'), n_imputed_days=i)
