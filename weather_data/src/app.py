import codecs
import csv
import sys
import urllib

import tables
from datetime import datetime
import requests
from time import sleep
import json
import os
import pymysql
import pandas as pd
from datetime import timedelta
import numpy as np
import time
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def connect_to_sql_database(
    db_server_name="localhost",
    db_user="axel",
    db_password="Mojjo@2015",
    db_name="south_africa",
    char_set="utf8mb4",
    cusror_type=pymysql.cursors.DictCursor,
):
    # print("connecting to db %s..." % db_name)
    global sql_db
    sql_db = pymysql.connect(
        host=db_server_name,
        user=db_user,
        password=db_password,
        db=db_name,
        charset=char_set,
        cursorclass=cusror_type,
    )
    return sql_db


def execute_sql_query(query, records=None, log_enabled=True):
    try:
        sql_db = connect_to_sql_database()
        cursor = sql_db.cursor()
        if records is not None:
            print("SQL Query: %s" % query, records)
            cursor.executemany(query, records)
        else:
            if log_enabled:
                print("SQL Query: %s" % query)
            cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            if log_enabled:
                print("SQL Answer: %s" % row)
        return rows
    except Exception as e:
        print("Exeception occured:{}".format(e))


def get_historical_weather_data(days_, out_file=None, city=None, farm_id=None):
    # days = execute_sql_query("SELECT DISTINCT timestamp_s FROM %s_resolution_day" % farm_id)
    # df = pd.read_csv(dataset_file, header=None)
    # days_ = pd.to_datetime(df[df.columns[-1]], format='%d/%m/%Y').dt.strftime('%Y-%m-%dT00:00')
    # days = pd.to_datetime(dates, format='%d/%m/%Y')
    # q_dates = []
    #
    # for d in days:
    #     for i in reversed(range(8)):
    #         p_d = d - timedelta(days=i)
    #         q_dates.append(p_d)
    #
    # days_ = np.array([x.strftime('%Y-%m-%dT00:00') for x in q_dates])
    # days_ = []
    # for item in days:
    #     days_.append(item['timestamp_s'].split(' ')[0])
    # h5 = tables.open_file(path, "r")
    # data = h5.root.resolution_d.data
    # days = []
    # for index, x in enumerate(data):
    #     # print("%d%%" % int((index/size)*100))
    #     ts = int(x['timestamp'])
    #     date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
    #     days.append(date)
    # days_ = list(set(days))
    # days_ = sorted(days_, key=lambda n: datetime.strptime(n, '%Y-%m-%d'))
    print(len(days_), days_)
    key = "3d109634f16f4f0ba27133341221410"
    URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    # purge_file(out_file)
    with open(out_file, "a") as outfile:
        for i, day in enumerate(days_):  # 2076
            PARAMS = {
                "key": key,
                "q": "%s,south+africa" % city,
                "date": day,
                "tp": 1,
                "format": "json",
            }
            r = requests.get(url=URL, params=PARAMS)
            data = r.json()
            print(data)
            print("progress %d/%d" % (i, len(days_)))
            json.dump(data, outfile)
            outfile.write("\n")
            sleep(0.2)

    print(outfile)


def format_time(time):
    if len(time) == 1:
        return "00:00"
    if len(time) == 3:
        return "0%s:00" % time.split("00")[0]
    if len(time) == 4 and time != "1000" and time != "2000":
        return "%s:00" % time.split("00")[0]
    if time == "1000":
        return "10:00"
    if time == "2000":
        return "20:00"


def get_humidity_date(path, name):
    data = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            js = json.loads(line)
            for w in js["data"]["weather"]:
                date = w["date"]
                list = []
                for h in w["hourly"]:
                    time = format_time(h["time"])
                    humidity = h["humidity"]
                    temp_c = h["tempC"]
                    list.append({"time": time, "humidity": humidity, "temp_c": temp_c})
                    print(date, time, temp_c, humidity)
                data[date] = list
        print(data)
        purge_file("%s_weather.json" % name)
        with open("%s_weather.json" % name, "a") as outfile:
            json.dump(data, outfile)


def create_weather_data_for_mrnn():
    df = pd.read_csv(
        "F:/Data2/mrnn_formatted_activity/cedara/sf_activity/activity_data.csv"
    )
    days = pd.to_datetime(df["date_str"], format="%Y-%m-%dT%H:%M")
    days_ = np.array([x.strftime("%Y-%m-%dT00:00") for x in days])
    days_ = np.unique(days_)

    get_historical_weather_data(days_, out_file="Cedara_weather.json", city="Cedara")
    data_dict = {}
    with open("Cedara_weather.json") as f:
        lines = f.readlines()
        for line in lines:
            js = json.loads(line)
            print(js)
            data_dict[js["data"]["weather"][0]["date"]] = js["data"]["weather"][0][
                "hourly"
            ]

    print(df)
    dates_q = np.array(
        [
            x.strftime("%Y-%m-%dT%H:%M")
            for x in pd.to_datetime(df["date_str"], format="%Y-%m-%dT%H:%M")
        ]
    )
    humidity_list = []
    temperature_list = []
    for d in dates_q:
        s = d.split("T")
        day_q = s[0]
        time_q = int(s[1].split(":")[0])
        data = data_dict[day_q][time_q]
        humidity = data["humidity"]
        temperature = data["tempC"]
        print(d, humidity, temperature)
        humidity_list.append(humidity)
        temperature_list.append(temperature)
    df["temperature"] = temperature_list
    df["humidity"] = humidity_list
    df.to_csv("F:/MRNN/data/cedara_activity_data_weather.csv", index=False)


def make_weather_calender(filepath, filename, title, start, end):
    print(filepath)
    with open(filepath) as f:
        lines = f.readlines()

        data = []
        for line in lines:
            js = json.loads(line)
            for w in js["data"]["weather"]:
                date = w["date"]
                for h in w["hourly"]:
                    time = format_time(h["time"])
                    humidity = int(h["humidity"])
                    temp_c = int(h["tempC"])
                    data.append(
                        {
                            "datetime": f"{date}T{time}",
                            "humidity": humidity,
                            "temp_c": temp_c,
                        }
                    )

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%dT%H:%M")

        df = df[df["datetime"] > pd.to_datetime(start, format="%B %Y")]

        df = df[df["datetime"] < pd.to_datetime(end, format="%B %Y")]

        df.index = df["datetime"]
        df = df.drop("datetime", 1)
        print(df)
        print(filename)
        df.to_csv(filename, index=False)

        df = df.sort_values(by="datetime")
        dfs = [group for _, group in df.groupby(df.index.strftime("%B/%Y"))]
        dfs = sorted(dfs, key=lambda x: x.index.max(axis=0))

        fig, axs = plt.subplots(
            3, int(np.ceil(len(dfs) / 3)), facecolor="white", figsize=(28.0, 10.80)
        )
        fig.suptitle(title, fontsize=14)
        axs = axs.ravel()
        for ax in axs:
            ax.set_axis_off()
        for i, d in enumerate(dfs):
            d.plot.area(
                ax=axs[i],
                rot=90,
                title=pd.to_datetime(d.index.values[0]).strftime("%B %Y"),
                ylim=(0, 100),
                stacked=False,
            )
            axs[i].set_axis_on()
        filepath = f"calender_{title}.png"
        print(filepath)
        fig.tight_layout()
        fig.savefig(filepath)


def create_weather_file(filepath, filepath2=None):
    print(filepath)
    df = pd.read_csv(filepath)

    if filepath2 is not None:
        df2 = pd.read_csv(filepath2).iloc[8760:, :]
        df = pd.concat([df, df2])

    # print(df)
    dfs = []
    for i in range(0, 24 * 365 * 10, 24):
        start = i
        end = i + 24
        # print(start, end)
        df_ = df.iloc[start:end, :][
            ["datetime", "temp", "humidity", "precip", "windspeed"]
        ]
        if df_.shape[0] == 0:
            break
        if df_.shape[0] > 24:
            raise ValueError("Thre should be 24 values in the dataframe!")
        dfs.append(df_)
    print(f"found {len(dfs)} days.")

    data = []
    for item in tqdm(dfs):
        date = pd.to_datetime(
            item["datetime"].values[0], format="%Y-%m-%dT%H:%M:%S"
        ).strftime("%d/%m/%Y")

        for index, row in item.iterrows():
            d_ = {
                "date": date,
                "datetime": row["datetime"],
                "humidity": row["humidity"],
                "temp_c": row["temp"],
                "precip": row["precip"],
                "windspeed": row["windspeed"],
            }
            data.append(d_)

    out_path = filepath.parent / f"{filepath.stem.replace(' ', '_')}.csv"
    print(out_path)
    df_data = pd.DataFrame(data)
    df_data.to_csv(out_path, index=False)


if __name__ == "__main__":
    print(sys.argv)

    filepath = (
        Path(os.path.dirname(os.path.dirname(__file__)))
        / "delmas south africa 2011-01-01 to 2015-12-31.csv"
    )

    filepath2 = (
        Path(os.path.dirname(os.path.dirname(__file__)))
        / "delmas south africa 2015-01-01 to 2017-01-01.csv"
    )

    create_weather_file(filepath, filepath2)

    filepath = (
        Path(os.path.dirname(os.path.dirname(__file__)))
        / "cedara south africa 2011-01-01 to 2015-12-31.csv"
    )
    create_weather_file(filepath)

    # make_weather_calender(Path("Delmas_weather.json"), "delmas_weather_data.csv", "Delmas Weather", "march 2015", "april 2016")
    # make_weather_calender(Path("Cedara_weather.json"), "cedara_weather_data.csv", "Cedara Weather", "june 2012", "july 2013")
    # create_weather_data_for_mrnn()
    # connect_to_sql_database()

    # start = datetime.datetime.strptime("01/01/2011", "%d/%m/%Y")
    # end = datetime.datetime.strptime("01/12/2015", "%d/%m/%Y")
    # date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    # days_ = np.array([x.strftime("%Y-%m-%d") for x in date_generated])

    # get_historical_weather_data(
    #     days_,
    #     out_file="delmas_weather_raw.json",
    #     farm_id="delmas_70101200027",
    #     city="Delmas"
    # )

    # start = datetime.datetime.strptime("01/01/2013", "%d/%m/%Y")
    # end = datetime.datetime.strptime("01/12/2013", "%d/%m/%Y")
    # date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    # days_ = np.array([x.strftime("%Y-%m-%d") for x in date_generated])

    # get_humidity_date('delmas_weather_raw.json', 'delmas')
    # get_historical_weather_data(
    #     days_,
    #     out_file="cedara_weather_raw.json",
    #     farm_id="cedara_70091100056",
    #     city="Cedara"
    # )
    #
    # # get_humidity_date('delmas_weather_raw.json', 'delmas')
    # get_humidity_date('cedara_weather_raw.json', 'cedara')
