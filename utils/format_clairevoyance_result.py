import datetime

import typer
from pathlib import Path
import pandas as pd
import numpy as np


def main(
    activity_data: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    output: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    )
):
    """This script reformats the clairevoyance results\n
    Args:\n
        activity_data: Folder containing clairevoyance imputed activity
        output: Output directory where the reformatted files will be created
    """
    print(activity_data)
    dfs = [g for _, g in pd.read_csv(activity_data).groupby(['id'])]
    for df in dfs:
        animal_id = int(df['id'].values[0])
        df = df[["time", "first_sensor_value", "imputed_feature_0"]]
        df["date_str"] = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M') for x in df["time"]]
        df.columns = ["timestamp", "first_sensor_value", "imputed", "date_str"]
        df = df[["timestamp", "date_str", "first_sensor_value", "imputed"]]
        filename = f"{animal_id}.csv"
        filepath = output / filename
        output.mkdir(parents=True, exist_ok=True)
        print(filepath)
        df.to_csv(filepath, index=False)


if __name__ == "__main__":
    typer.run(main)
