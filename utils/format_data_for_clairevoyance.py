import typer
from pathlib import Path
import pandas as pd


def format(df, id):
    df_static = df[["first_sensor_value"]]
    df_static.insert(0, "id", id)
    df_static.columns = ['id', 'first_sensor_value']

    df_temporal = df[["timestamp", "first_sensor_value"]]
    df_temporal.insert(0, "id", id)
    df_temporal.insert(2, "variable", "first_sensor_value")
    df_temporal.columns = ['id', 'time', "variable", "value"]

    return df_static, df_temporal


def main(
    dataset: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    )
):
    print(dataset)
    files = dataset.glob('*.csv')
    df_static_list = []
    df_temporal_list = []
    for file in files:
        id = file.stem
        print(f"animal id: {id}")
        df = pd.read_csv(file)
        df_static, df_temporal = format(df, id)
        df_static_list.append(df_static)
        df_temporal_list.append(df_temporal)
    df_static_f = pd.concat(df_static_list, axis=0)
    df_temporal_f = pd.concat(df_temporal_list, axis=0)
    print(f"df_static_f: {df_static_f}")
    print(f"df_temporal_f: {df_temporal_f}")


if __name__ == "__main__":
    typer.run(main)
