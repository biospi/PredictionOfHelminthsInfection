import typer
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

from model.data_loader import load_activity_data
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import plot_umap, plot_time_pca

IDS = [
    "'Greg'",
    "'Henry'",
    "'Tilly'",
    # "'Maisie'",
    "'Sookie'",
    # "'Oliver_F'",
    "'Ra'",
    "'Hector'",
    "'Jimmy'",
    # "'MrDudley'",
    "'Kira'",
    # "'Lucy'",
    "'Louis'",
    "'Luna_M'",
    "'Wookey'",
    "'Logan'",
    "'Ruby'",
    "'Kobe'",
    "'Saffy_J'",
    "'Enzo'",
    "'Milo'",
    "'Luna_F'",
    "'Oscar'",
    "'Kia'",
    "'Cat'",
    "'AlfieTickles'",
    "'Phoebe'",
    "'Harvey'",
    "'Mia'",
    "'Amadeus'",
    "'Marley'",
    "'Loulou'",
    "'Bumble'",
    "'Skittle'",
    "'Charlie_O'",
    "'Ginger'",
    "'Hugo_M'",
    "'Flip'",
    "'Guinness'",
    "'Chloe'",
    "'Bobby'",
    "'QueenPurr'",
    "'Jinx'",
    "'Charlie_B'",
    "'Thomas'",
    "'Sam'",
    "'Max'",
    "'Oliver_S'",
    "'Millie'",
    "'Clover'",
    "'Bobbie'",
    "'Gregory'",
    "'Kiki'",
    "'Hugo_R'",
    "'Shadow'",
]

IDS2 = [
    "Mia",
    "Loulou",
    "Sam",
    "Enzo",
    "Amadeus",
    "Bobbie",
    "Kobe",
    "Hugo_R",
    "Wookey",
    "Millie",
]


def main(
    dataset_file: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    ),
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    ids: List[str] = [],
    ids_g1: List[str] = [],
    ids_g2: List[str] = [],
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
        "age",
        "name",
        "mobility_score",
    ],
    preprocessing_steps: List[str] = ["QN"],
    class_healthy_label: List[str] = ["0.0"],
    class_unhealthy_label: List[str] = ["1.0"],
    individual_to_ignore: List[str] = ["MrDudley", "Oliver_F", "Lucy"],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ids={ids}")
    print(f"loading dataset file {dataset_file} ...")
    (
        data_frame,
        meta_data,
        meta_data_short,
        _,
        _,
        label_series,
        samples,
        seasons_features,
    ) = load_activity_data(
        output_dir,
        meta_columns,
        dataset_file,
        -1,
        class_healthy_label,
        class_unhealthy_label,
        imputed_days=-1,
        preprocessing_steps=preprocessing_steps,
        individual_to_ignore=individual_to_ignore,
        meta_cols_str=[],
        sampling="T",
        individual_to_keep=[],
        resolution=None,
    )

    data_frame_time, _, _ = apply_preprocessing_steps(
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
        keep_meta=True,
    )
    print(data_frame_time)

    for id in ids:
        id = id.strip().replace("'", "")
        A = data_frame_time[data_frame_time["name"] == id]
        make_plot(A, output_dir, id, len(meta_columns))
        # sub = "bad"
        # if id in ids_g1:
        #     sub = "good"
        # make_plot(A, output_dir /sub, id)

    # A1 = df[df["id"].isin(ids_g1)]
    # make_plot(A1, output_dir, "group on the right")
    # A2 = df[df["id"].isin(ids_g2)]
    # make_plot(A2, output_dir, "group on the left")

    data_frame_time = data_frame_time[data_frame_time["name"].isin(ids)]
    plot_time_pca(
        len(meta_columns),
        data_frame_time.copy(),
        output_dir / "pca",
        label_series,
        y_col="id",
        title="PCA time domain after normalisation",
    )


    plot_umap(
        meta_columns,
        data_frame_time.copy(),
        output_dir / "umap",
        label_series,
        y_col="name",
        title="UMAP time domain after normalisation",
    )


def make_plot(A, output_dir, id, n_meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    # id='_'.join(id).replace("'",'')
    health = A["health"].mean()
    df_activity = A.iloc[:, :-n_meta]
    print(df_activity)
    title = f"Samples after normalisation mean health={health:.2f}"
    plt.clf()
    fig = df_activity.T.plot(
        kind="line",
        subplots=False,
        grid=True,
        legend=False,
        title=title,
        alpha=0.7,
        xlabel="Time(s)",
        ylabel="Activity count",
    ).get_figure()
    plt.ylim(0, 70)
    plt.tight_layout()
    filepath = output_dir / f"{title}.png"
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    main(
        dataset_file=Path(
            "E:/Cats/build_permutations/1000__003__0_00100__120/dataset/training_sets/samples/samples.csv"
        ),
        output_dir=Path("E:/Cats/build_permutations/visu"),
        ids=IDS2,
    )
