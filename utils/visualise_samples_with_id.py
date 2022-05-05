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
    "'Maisie'",
    "'Sookie'",
    "'Oliver_F'",
    "'Ra'",
    "'Hector'",
    "'Jimmy'",
    "'MrDudley'",
    "'Kira'",
    "'Lucy'",
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
        health = A["health"].values[0]
        df_activity = A.iloc[:, :-8]
        print(df_activity)
        title = f"Samples for {id} health={health}"
        fig = df_activity.T.plot(
            kind="line",
            subplots=True,
            grid=True,
            title=title,
            xlabel="Time",
            ylabel="Activity",
            layout=(len(df_activity), 1),
            sharex=True,
            sharey=False,
            legend=False,
            figsize=(10, 20),
        ).ravel()[0].get_figure()
        plt.tight_layout()
        filepath = output_dir / f"{title}.png"
        print(filepath)
        fig.savefig(filepath)


if __name__ == "__main__":
    main(
        dataset_file=Path(
            "E:/Cats/build_multiple_peak_2/002__0_00100__120/dataset/training_sets/samples/samples.csv"
        ),
        output_dir=Path("E:/Cats/build_multiple_peak_2/visu")

    )
