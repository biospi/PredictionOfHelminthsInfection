import json

import typer
from pathlib import Path
import pandas as pd
import numpy as np
import gzip

from tqdm import tqdm

from model.data_loader import load_activity_data
from utils._anscombe import anscombe
from sys import exit
from datetime import datetime


def build_gt_data(df_famacha, df_temporal):
    df_temporal_ = df_temporal.copy()
    df_temporal_["datetime"] = pd.to_datetime(
        df_temporal_["time"], unit="s"
    )
    df_temporal_["date"] = pd.to_datetime(
        df_temporal_["time"], unit="s"
    ).dt.strftime("%d/%m/%Y")

    df_temporal_["variable"] = "famacha"
    values = np.zeros(df_temporal_.shape[0])
    values[:] = np.nan
    df_temporal_["value"] = values

    df_famacha["target"] = (df_famacha["target"].values != 1).astype(int)
    famacha_map = dict(zip(df_famacha["date"].values, df_famacha["target"].values))

    df_temporal_ = df_temporal_.replace({"date": famacha_map})

    df_temporal_["value"] = [x if len(str(x)) == 1 else np.nan for x in df_temporal_["date"].values]

    df_temporal_.drop(['date', 'datetime'], axis=1, inplace=True)

    return df_temporal_


def combine_dfs(df1, df2):
    temp = pd.DataFrame(np.zeros((df2.shape[0]*2, df2.shape[1])), columns=df1.columns)
    idx1 = np.arange(0, temp.shape[0], 2)
    idx2 = np.arange(1, temp.shape[0], 2)
    df1.index = idx1
    df2.index = idx2
    temp.iloc[idx1, :] = df1
    temp.iloc[idx2, :] = df2
    return temp


def format(df, farm_id, id, ground_truth_file):
    #df["first_sensor_value"] = [anscombe(np.array([x]))[0] for x in df["first_sensor_value"]]
    df_static = df[["first_sensor_value"]]
    df_static.insert(0, "id", farm_id)
    df_static.columns = ["id", "first_sensor_value"]

    df_temporal = df[["timestamp", "first_sensor_value"]]
    df_temporal.insert(0, "id", farm_id)
    df_temporal.insert(2, "variable", f"first_sensor_value_{id}")
    df_temporal.columns = ["id", "time", "variable", "value"]

    df_treatment = build_gt_data(ground_truth_file, df_temporal)

    # df_treatment2 = df_treatment.copy()
    # df_treatment2["variable"] = "mock_feature1"
    # df_treatment2["value"] = df_temporal["value"]
    #
    # df_treatment3 = df_treatment.copy()
    # df_treatment3["variable"] = "mock_feature2"
    # df_treatment3["value"] = df_temporal["value"]

    df_temporal_ground = combine_dfs(df_temporal, df_treatment)
    # df_temporal_ground = []
    # for (index1, row1), (index2, row2) in zip(
    #     df_temporal.iterrows(), df_treatment.iterrows()
    # ):
    #     #skip rows with missing famacha
    #     if np.isnan(row2["value"]):
    #         continue
    #     df_temporal_ground.append(row1.values)
    #     df_temporal_ground.append(row2.values)
    #     # df_temporal_ground.append(row3.values)
    #     # df_temporal_ground.append(row4.values)
    # df_temporal_ground = pd.DataFrame(df_temporal_ground)
    # df_temporal_ground.columns = df_temporal.columns

    return df_static, df_temporal_ground


def create_archive(file_path):
    print(f"saving file in {file_path} ...")
    input_path = str(file_path)
    output_path = str(file_path.parent / f"{file_path.name}.gz")
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
        f_out.writelines(f_in)


def split_dataframe(df, chunk_size=1440):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def filter_empty_days(df, thresh):
    dfs = split_dataframe(df, chunk_size=1440)
    filtered = []
    for d in dfs:
        if d.shape[0] != 1440:
            continue
        nan_count = sum(np.isnan(d["first_sensor_value"]).astype(int))
        if nan_count < thresh:
            continue
        filtered.append(d)
    if len(filtered) == 0:
        return None
    df_filtered = pd.concat(filtered)
    return df_filtered


def get_weather_data(timestamps, w_d):
    humidity = np.zeros(len(timestamps))
    humidity[:] = np.nan
    temperature = np.zeros(len(timestamps))
    temperature[:] = np.nan
    for i, epoch in enumerate(timestamps):
        date_str = datetime.fromtimestamp(epoch).strftime('%Y-%m-%d')
        if date_str in w_d.keys():
            data = w_d[date_str]
            date_str_time = datetime.fromtimestamp(epoch).strftime('%H')
            for d in data:
                t = d["time"].split(':')[0]
                if t == date_str_time:
                    hum = d["humidity"]
                    temp = d["temp_c"]
                    humidity[i] = hum
                    temperature[i] = temp
                    break
    return humidity, temperature


def main(
    activity_data: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_ground: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    output: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    weather_file: Path = None,
    split: int = 20,
    thresh: int = 100,
    thresh_nan: int = 100,
    farm_id: int = 0,
    data_name: str = "sf_activity",
):
    """This script reformats the raw backfilled activity data for clairevoyance imputation\n
    Args:\n
        activity_data: Folder containing the backfilled .csv files
        dataset_ground:  File containing famacha data
        output: Output directory where the reformatted files will be created
        split: Test size in percent
        thresh: minimum activity points that need to be in backfilled file
    """

    weather_data = None
    if weather_file is not None:
        with open(weather_file) as json_file:
            weather_data = json.load(json_file)

    print(activity_data)
    files = list(activity_data.glob("*.csv"))
    df_static_list = []
    df_temporal_list = []

    day = int(dataset_ground.stem.split("dbft_")[1][0])
    (
        df_famacha,
        N_META,
        class_healthy_target,
        class_unhealthy_target,
        label_series,
    ) = load_activity_data(str(dataset_ground), day, "1To1", "2To2")

    activity = []
    ids = []
    w_d = None
    for i, file in enumerate(files):
        print(f"progress {i}/{len(files)}...")
        id = file.stem
        df = pd.read_csv(file)

        if w_d is None:
            w_d = get_weather_data(df["timestamp"].values, weather_data)

        tot_activity = df["first_sensor_value"].values.astype(float)
        mask = ~np.isnan(tot_activity)
        nan_record_count = np.sum(mask.astype(int))

        zeros_record_count = tot_activity[tot_activity == 0]

        if nan_record_count < thresh or len(zeros_record_count) < thresh:
            print(
                f"very few ({nan_record_count}/{df.shape[0]}) valid activity counts in file. dismiss file."
            )
            continue
        print(f"animal id: {id}")
        #df = filter_empty_days(df, thresh_nan)
        # if df is None:
        #     continue
        #df = df[126117:126117+1440*7]
        activity.append(df["first_sensor_value"])
        ids.append(id)
        df_static, df_temporal = format(df, farm_id, id, df_famacha)
        df_static_list.append(df_static)
        df_temporal_list.append(df_temporal)
        # if len(df_temporal_list) > 2:
        #     break

    if w_d is not None:
        activity.append(pd.Series(w_d[0]))
        activity.append(pd.Series(w_d[1]))
        ids.append("humidity")
        ids.append("temperature")

    df_activity = pd.concat(activity, axis=1)
    df_activity.columns = ids

    output = output / data_name
    output.mkdir(parents=True, exist_ok=True)

    df_activity["timestamp"] = df["timestamp"]
    df_activity["date_str"] = df["date_str"]
    path = output / "activity_data.csv"
    print(path)
    df_activity.to_csv(path, index=False)

    df_static_f = pd.concat(df_static_list, axis=0)
    df_temporal_f = pd.concat(df_temporal_list, axis=0)

    df_temporal_f = df_temporal_f.sort_values("time").drop_duplicates()

    #covert timestamp epoch to integers
    # s = pd.Series(df_temporal_f["time"].to_list())
    # hot = pd.get_dummies(s).values
    # time = hot*np.arange(0, hot.shape[1], 1)
    # time = time.sum(axis=1)
    #
    # df_temporal_f["time"] = time

    print(f"df_static_f:\n {df_static_f}")
    print(f"df_temporal_f:\n {df_temporal_f}")

    test_size_static = int(split / 100 * df_static_f.shape[0])
    train_size_static = df_static_f.shape[0] - test_size_static
    print(f"train size static: {train_size_static} test size: {test_size_static}")

    test_size_temporal = int(split / 100 * df_temporal_f.shape[0])
    train_size_temporal = df_temporal_f.shape[0] - test_size_temporal
    print(f"train size temporal: {train_size_temporal} test size: {test_size_temporal}")

    static_test_data = df_static_f.iloc[:, :]
    static_train_data = df_static_f.iloc[:, :]
    temporal_test_data = df_temporal_f.iloc[:, :]
    temporal_train_data = df_temporal_f.iloc[:, :]

    path = output / f"{data_name}_static_test_data.csv"
    static_test_data.to_csv(path, index=False)
    create_archive(path)

    path = output / f"{data_name}_static_train_data.csv"
    static_train_data.to_csv(path, index=False)
    create_archive(path)

    path = output / f"{data_name}_temporal_test_data_eav.csv"
    temporal_test_data.to_csv(path, index=False)
    create_archive(path)

    path = output / f"{data_name}_temporal_train_data_eav.csv"
    temporal_train_data.to_csv(path, index=False)
    create_archive(path)


if __name__ == "__main__":
    typer.run(main)
