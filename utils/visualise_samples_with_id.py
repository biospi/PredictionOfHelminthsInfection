import typer
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt


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


def main(
    dataset_file: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    ),
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    ids: List[str] = [
        "'Loulou'", "'Enzo'", "'Oscar'", "'Maisie'", "'Millie'"
    ],
    # ids_g1: List[str] = [
    #     "'Chloe'", "'Wookey'", "'Max'", "'Gregory'", "'Kia'", "'Bumble'", "'Sookie'", "'Flip'", "'Kobe'", "'Ra'", "'Hector'", "'Jinx'", "'Hugo_M'", "'Henry'", "'Shadow'", "'Cat'", "'Saffy_l'", "'Kiki'", "'Thomas'", "'Milo'", "'Bobby'", "'Luna_F'", "'QueenPurr'", "'Oliver_S'", "'Phoebe'", "'Bobbie'", "'Amadeus'", "'Tilly'", "'Luna_M'"
    # ],
    ids_g1: List[str] = [
        "'Chloe'", "'Wookey'", "'Max'", "'Gregory'", "'Kia'", "'Bumble'", "'Sookie'", "'Flip'", "'Kobe'", "'Ra'",
        "'Hector'", "'Jinx'", "'Hugo_M'", "'Henry'", "'Shadow'", "'Cat'", "'Saffy_l'", "'Kiki'", "'Thomas'",
        "'Milo'", "'Bobby'"
    ],
    ids_g2: List[str] = ["'Greg'", "'Jimmy'", "'Kira'", "'Louis'", "'Logan'", "'Ruby'", "'Saffy_J'", "'Enzo'", "'Oscar'", "'AlfieTickles'", "'Harvey'", "'Mia'", "'Marley'", "'Loulou'", "'Skittle'", "'Charlie_O'", "'Ginger'", "'Guinness'", "'Charlie_B'", "'Sam'", "'Millie'", "'Clover'", "'Hugo_R'"]

):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ids={ids}")
    df = pd.read_csv(dataset_file, header=None)
    columns = list(df.columns)
    columns[-1] = "health"
    columns[-2] = "id"
    df.columns = columns

    for id in ids:
        A = df[df["id"] == id]
        sub = "bad"
        if id in ids_g1:
            sub = "good"
        make_plot(A, output_dir /sub, id)

    A1 = df[df["id"].isin(ids_g1)]
    make_plot(A1, output_dir, "group on the right")
    A2 = df[df["id"].isin(ids_g2)]
    make_plot(A2, output_dir, "group on the left")


def make_plot(A, output_dir, id):
    output_dir.mkdir(parents=True, exist_ok=True)
    #id='_'.join(id).replace("'",'')
    health = A["health"].mean()
    df_activity = A.iloc[:, :-8]
    print(df_activity)
    title = f"Samples for {id} mean health={health:.2f}"
    plt.clf()
    fig = df_activity.T.plot(
        kind="line",
        subplots=False,
        grid=True,
        legend=False,
        title=title,
        alpha=0.7,
        xlabel="Time",
        ylabel="Activity"
    ).get_figure()
    plt.ylim(0, 1500)
    plt.tight_layout()
    filepath = output_dir / f"{title}.png"
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    main(
        dataset_file=Path(
            "E:/Cats/build_multiple_peak_2/002__0_00100__120/dataset/training_sets/samples/samples.csv"
        ),
        output_dir=Path("E:/Cats/build_multiple_peak_2/visu2"),
        ids=IDS
    )
