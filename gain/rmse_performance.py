import typer
from pathlib import Path
import pandas as pd
import glob
import numpy as np


def rmse(original: np.numarray, imputed: np.numarray):
    diff_gain = original.values - imputed.values
    nominator = np.nansum(diff_gain ** 2)
    denominator = diff_gain[~np.isnan(diff_gain)].size
    print("nominator =", nominator)
    print("denominator =", denominator)
    rmse = np.sqrt(nominator / float(denominator))
    return rmse


def main(
    gain_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    mrnn_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
):
    gain_files = glob.glob(f'{gain_dir}/*/**.csv')
    mrnn_files = glob.glob(f'{mrnn_dir}/*/**.csv')
    df_gain = pd.DataFrame()
    for g_f in gain_files:
        if "miss_rate_0_2_" in g_f:
            print(g_f)
            df = pd.read_csv(g_f)["first_sensor_value_gain"]
            df_gain[Path(g_f).stem] = df
    df_mrnn = pd.DataFrame()
    df_li = pd.DataFrame()
    df_original = pd.DataFrame()
    for g_mrnn in mrnn_files:
        if "3_missingrate_[0.2]_seql_1440_iteration_100" in g_mrnn:
            print(g_mrnn)
            id = Path(g_mrnn).stem
            first_sensor_value_mrnn = pd.read_csv(g_mrnn)['first_sensor_value_mrnn']
            df_mrnn[id] = first_sensor_value_mrnn

            first_sensor_value = pd.read_csv(g_mrnn)['first_sensor_value']
            df_original[id] = first_sensor_value

            first_sensor_value_li = pd.read_csv(g_mrnn)['first_sensor_value_li']
            df_li[id] = first_sensor_value_li

    #print(output_dir)

    df_mrnn = df_mrnn[df_gain.columns].iloc[0:df_gain.shape[0], :]
    df_li = df_li[df_gain.columns].iloc[0:df_gain.shape[0], :]
    df_gain = df_gain[df_gain.columns].iloc[0:df_gain.shape[0], :]
    df_original = df_original[df_gain.columns].iloc[0:df_gain.shape[0], :]

    print(df_gain.shape, df_mrnn.shape, df_li.shape, df_original.shape)
    rmse_gain = rmse(df_original, df_gain)
    rmse_mrnn = rmse(df_original, df_mrnn)
    rmse_li = rmse(df_original, df_li)

    print(rmse_gain, rmse_mrnn, rmse_li)


if __name__ == "__main__":
    typer.run(main)
