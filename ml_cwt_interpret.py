import glob
from pathlib import Path
from typing import List

import pandas as pd
import typer

from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import applyPreprocessingSteps


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_folder: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    class_healthy_label: str = "1To1",
    class_unhealthy_label: str = "2To2",
    steps: List[str] = ["QN", "ANSCOMBE", "LOG", "CWT(MEXH)"],
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
            n_scales=None,
            farm_name="FARMS",
            keep_meta=False,
        )
        print(data_frame)


if __name__ == "__main__":
    typer.run(main)