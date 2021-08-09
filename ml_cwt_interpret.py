import glob
from pathlib import Path
from typing import List

import typer

from classifier.src.cwt_weight import (
    chunck_df,
    explain_cwt,
    get_cwt_data_frame,
    process_df, plot_ribbon)
from model.data_loader import load_activity_data, parse_param_from_filename
from preprocessing.preprocessing import applyPreprocessingSteps
import pandas as pd


def plot_progression(output_dir, days):
    print("plot progression...")
    files = [x for x in glob.glob(str(output_dir / "RepeatedKFold" / "*.csv")) if "rbf" in x]
    aucs = {}
    for i, file in enumerate(files):
        df = pd.read_csv(str(file), converters={'roc_auc_scores': eval})
        a = df["roc_auc_scores"][0]
        aucs[i] = a

    plot_ribbon(
        output_dir,
        aucs,
        "Classifier Auc over time during increase of the FAMACHA score",
        "Auc",
        days,
    )


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
    p: bool = typer.Option(False, "--p")
):
    """This script builds the graphs for cwt interpretation\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
        p: analyse famacha impact over time up to test date
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

        df_cwt, class0_count, class1_count, cwt_coefs_data = get_cwt_data_frame(data_frame)
        if p:
            dfs, data = chunck_df(days, df_cwt, cwt_coefs_data)
        else:
            dfs, data = process_df(df_cwt, cwt_coefs_data)

        explain_cwt(
            days,
            dfs,
            data,
            output_dir,
            class0_count,
            class1_count
        )

        if p:
            plot_progression(output_dir, days)


if __name__ == "__main__":
    typer.run(main)
