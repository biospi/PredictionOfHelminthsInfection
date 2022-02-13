import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from model.svm import make_roc_curve, process_clf
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.Utils import getXY, plot_heatmap


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    n_imputed_days: int = 7,
    n_activity_days: int = 7,
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    meta_columns: List[str] = ["label", "id", "imputed_days", "date", "health"],
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG"],
    train_size: float = 0.9,
    n_fold: int = 50
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

        # data_frame = data_frame[
        #     data_frame["health"].isin([0, 1])
        # ]

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
            meta_columns,
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

        plot_heatmap(X1, y1, X2, y2, output_dir, p1_start, p1_end, p2_start, p2_end)

        process_clf(train_size, label_series, label_series, info, preprocessing_steps, n_fold, X1, X2, y1, y2, output_dir)
        process_clf(train_size, label_series, label_series, info, preprocessing_steps, n_fold, X2, X1, y2, y1, output_dir / 'rev')


if __name__ == "__main__":
    #typer.run(main)

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        main(Path(f'E:/Data2/debug/temporal/{i}'), Path('E:/Data2/debug/delmas/dataset_mrnn_7day'), imputed_days=i)

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        main(Path(f'E:/Data2/debug/temporal/{i}'), Path('E:/Data2/debug3/cedara/dataset6_mrnn_7day'), imputed_days=i,
             class_unhealthy_label=["2To2", "2To4", "3To4", "1To4", "1To3", "4To5", "2To3"],)
