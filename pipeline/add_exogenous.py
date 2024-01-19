import json
import sys
from pathlib import Path
from sys import exit
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
from utils.Utils import create_rec_dir


def get_temp(d_dates, weather_file_path, metric="temp_c"):
    print("get temp...")
    with open(weather_file_path) as json_file:
        data = json.load(json_file)
    print("loaded weather data")
    exo = []
    for row in tqdm(d_dates):
        temp_ = []
        #print(row)
        for item in row:
            empty_day_data = np.ones(1440)
            empty_day_data[:] = np.nan
            df_row_data = pd.DataFrame(empty_day_data)
            time_range = pd.date_range(start="2018-09-09", end="2018-09-10", freq="T").strftime("%H:%M").tolist()[:-1]
            df_row_data.index = time_range
            key = item.strftime("%Y-%m-%d")
            if key not in data:
                print("missing data for date=", key)
                temp_.append(df_row_data[0].values)
                continue
            for elem in data[key]:
                v = elem[metric]
                df_row_data.loc[elem["time"]] = v
            df_row_data_i = pd.Series(df_row_data[0].astype(float).values).interpolate(method='nearest').ffill().bfill().values
            temp_.append(df_row_data_i)
        row_data = np.array(temp_).flatten()
        exo.append(row_data)

    data_exo = pd.DataFrame(np.array(exo))
    print(data_exo)
    return data_exo


def main(dataset, out_dir, weather_file):
    dataset = Path(dataset)
    out_dir = Path(out_dir)
    weather_file = Path(weather_file)

    print("dataset=", dataset)
    print("outDir=", out_dir)
    df = pd.read_csv(dataset, header=None)
    header = df.columns.tolist()
    header[-1] = "label"
    header[-3] = "id"
    header[-2] = "missingness"
    header[-1] = "date"
    df.columns = header
    print(df)
    dates = df["date"]
    dates = pd.to_datetime(dates, format="%d/%m/%Y").values
    n_days = int(dataset.name.split("_")[-2])

    data = []
    for d in dates:
        q_dates = []
        for i in reversed(range(n_days)):
            p_d = pd.to_datetime(d) - timedelta(days=i)
            q_dates.append(p_d)
        data.append(q_dates)

    df_temp = get_temp(np.array(data), weather_file, metric="temp_c")
    df_humidity = get_temp(np.array(data), weather_file, metric="humidity")

    df_concat_temp_df = pd.concat([df_temp], axis=1)
    df_concat_humidity_df = pd.concat([df_humidity], axis=1)
    df_concat_temp_humidity_df = pd.concat([df_temp, df_humidity], axis=1)

    out_dir_base = str(out_dir).replace("\\", "/")
    out_dir_temp = out_dir_base+"/temp"
    create_rec_dir(out_dir_temp)

    filename = "%s/temp_%s" % (out_dir_temp, dataset.name)
    print(filename)
    df_concat_temp_df.to_csv(filename, index=False)
    t_f = filename

    out_dir_humidity = out_dir_base+"/humidity"
    create_rec_dir(out_dir_humidity)
    filename = "%s/humidity_%s" % (out_dir_humidity, dataset.name)
    print(filename)
    df_concat_humidity_df.to_csv(filename, index=False)
    h_f = filename

    out_dir_temp_humidity = out_dir_base+"/temp_humidity"
    create_rec_dir(out_dir_temp_humidity)
    filename = "%s/temp_humidity_%s" % (out_dir_temp_humidity, dataset.name)
    print(filename)
    df_concat_temp_humidity_df.to_csv(filename, index=False)
    return t_f, h_f


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: "
              "add_exogenous <Dataset> <Output Directory> <Weather File>")
        exit(1)

    dataset = sys.argv[1]
    out_dir = sys.argv[2]
    weather_file = sys.argv[3]

    main(dataset, out_dir, weather_file)

