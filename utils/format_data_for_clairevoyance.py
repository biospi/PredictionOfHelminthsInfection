import typer
from pathlib import Path
import pandas as pd
import numpy as np


def format(df, id):
    df_static = df[["first_sensor_value"]]
    df_static.insert(0, "id", id)
    df_static.columns = ["id", "first_sensor_value"]

    df_temporal = df[["timestamp", "first_sensor_value"]]
    df_temporal.insert(0, "id", id)
    df_temporal.insert(2, "variable", "first_sensor_value")
    df_temporal.columns = ["id", "time", "variable", "value"]

    return df_static, df_temporal


def main(
    dataset: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    output: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    split: int = 20,
    thresh: int = 100,
):
    """This script reformats the raw backfilled activity data for clairevoyance imputation\n
    Args:\n
        dataset: Folder containing the backfilled .csv files
        output: Output directory where the reformatted files will be created
        split: Test size in percent
        thresh: minimum activity points that need to be in backfilled file
    """
    print(dataset)
    files = dataset.glob("*.csv")
    df_static_list = []
    df_temporal_list = []
    for file in files:
        id = file.stem
        df = pd.read_csv(file)
        tot_activity = df["first_sensor_value"].values.astype(float)
        mask = ~np.isnan(tot_activity)
        valid_record_count = np.sum(mask.astype(int))
        if valid_record_count < thresh:
            print(f"very few ({valid_record_count}/{df.shape[0]}) valid activity counts in file. dismiss file.")
            continue
        print(f"animal id: {id}")
        df_static, df_temporal = format(df, id)
        df_static_list.append(df_static)
        df_temporal_list.append(df_temporal)

    df_static_f = pd.concat(df_static_list, axis=0)
    df_temporal_f = pd.concat(df_temporal_list, axis=0)
    print(f"df_static_f:\n {df_static_f}")
    print(f"df_temporal_f:\n {df_temporal_f}")

    test_size = int(split / 100 * df_static_f.shape[0])
    train_size = df_static_f.shape[0] - test_size
    print(f"train size: {train_size} test size: {test_size}")

    static_test_data = df_static_f.iloc[0:test_size, :]
    static_train_data = df_static_f.iloc[test_size: train_size, :]
    temporal_test_data = df_temporal_f.iloc[0: test_size, :]
    temporal_train_data = df_temporal_f.iloc[test_size: train_size, :]

    output.mkdir(parents=True, exist_ok=True)
    print(f"saving file in {output} ...")
    static_test_data.to_csv(output / "static_test_data.csv", index=False)
    print(f"saving file in {output} ...")
    static_train_data.to_csv(output / "static_train_data.csv", index=False)
    print(f"saving file in {output} ...")
    temporal_test_data.to_csv(output / "temporal_test_data.csv", index=False)
    print(f"saving file in {output} ...")
    temporal_train_data.to_csv(output / "temporal_train_data.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
