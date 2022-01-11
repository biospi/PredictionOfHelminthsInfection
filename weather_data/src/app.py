import sys
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


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def connect_to_sql_database(db_server_name="localhost", db_user="axel", db_password="Mojjo@2015", db_name="south_africa",
                            char_set="utf8mb4", cusror_type=pymysql.cursors.DictCursor):
    # print("connecting to db %s..." % db_name)
    global sql_db
    sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password,
                             db=db_name, charset=char_set, cursorclass=cusror_type)
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


def get_historical_weather_data(days_, out_file=None, city=None):
    # days = execute_sql_query("SELECT DISTINCT timestamp_s FROM %s_resolution_day" % farm_id)
    #df = pd.read_csv(dataset_file, header=None)
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
    REQ = 499
    keys = ["8d6a9b03cf40436e9f203817212411"]
    n_d = np.ceil(len(days_)/REQ)
    print("For 500 req a day you'll need %d days." % n_d)
    URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    purge_file(out_file)
    with open(out_file, 'a') as outfile:
        i = 0
        while True:
            if i >= len(days_):
                break

            for key in keys:
                cpt = 0

                while True:
                    if i >= len(days_):
                        break

                    PARAMS = {'key': key, 'q': "%s,south+africa" % city,
                              'date': days_[i], 'tp': 1, 'format': 'json'}
                    r = requests.get(url=URL, params=PARAMS)
                    data = r.json()
                    print("progress %d/%d" % (i, len(days_)))
                    json.dump(data, outfile)
                    outfile.write('\n')
                    sleep(0.5)
                    cpt += 1
                    i += 1
                    if cpt >= REQ:
                        print("use next avail key %s." % key)
                        break

            print("used all avail keys wait until 2AM...")
            t = datetime.datetime.today()
            future = datetime.datetime(t.year, t.month, t.day, 2, 0)
            if t.hour >= 2:
                future += datetime.timedelta(days=1)
            time.sleep((future - t).total_seconds())
            print("reached max req %d for the day. Wait for next day..." % REQ*len(keys))

    print(outfile)


def format_time(time):
    if len(time) == 1:
        return '00:00'
    if len(time) == 3:
        return "0%s:00" % time.split('00')[0]
    if len(time) == 4 and time != '1000' and time != '2000':
        return "%s:00" % time.split('00')[0]
    if time == '1000':
        return '10:00'
    if time == '2000':
        return '20:00'


def get_humidity_date(path, name):
    data = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            js = json.loads(line)
            for w in js['data']['weather']:
                date = w['date']
                list = []
                for h in w['hourly']:
                    time = format_time(h['time'])
                    humidity = h['humidity']
                    temp_c = h['tempC']
                    list.append({'time': time, 'humidity': humidity, 'temp_c': temp_c})
                    print(date, time, temp_c, humidity)
                data[date] = list
        print(data)
        purge_file('%s_weather.json' % name)
        with open('%s_weather.json' % name, 'a') as outfile:
            json.dump(data, outfile)


def create_weather_data_for_mrnn():
    df = pd.read_csv("F:/Data2/mrnn_formatted_activity/cedara/sf_activity/activity_data.csv")
    days = pd.to_datetime(df["date_str"], format='%Y-%m-%dT%H:%M')

    days_ = np.array([x.strftime('%Y-%m-%dT00:00') for x in days])
    days_ = np.unique(days_)

    get_historical_weather_data(days_, out_file="Cedara_weather.json", city="Cedara")
    data_dict = {}
    with open("Cedara_weather.json") as f:
        lines = f.readlines()
        for line in lines:
            js = json.loads(line)
            print(js)
            data_dict[js['data']['weather'][0]['date']] = js['data']['weather'][0]['hourly']

    print(df)
    dates_q = np.array([x.strftime('%Y-%m-%dT%H:%M') for x in pd.to_datetime(df["date_str"], format='%Y-%m-%dT%H:%M')])
    humidity_list = []
    temperature_list = []
    for d in dates_q:
        s = d.split('T')
        day_q = s[0]
        time_q = int(s[1].split(':')[0])
        data = data_dict[day_q][time_q]
        humidity = data['humidity']
        temperature = data['tempC']
        print(d, humidity, temperature)
        humidity_list.append(humidity)
        temperature_list.append(temperature)
    df["temperature"] = temperature_list
    df["humidity"] = humidity_list
    df.to_csv("F:/MRNN/data/cedara_activity_data_weather.csv", index=False)


if __name__ == '__main__':
    print(sys.argv)
    create_weather_data_for_mrnn()
    #connect_to_sql_database()
    #get_historical_weather_data("F:/Data2/dataset_gain_7day/activity_delmas_70101200027_dbft_7_1min.csv", farm_id="delmas_70101200027", out_file="delmas_weather_raw.json", city="Delmas")
    # get_humidity_date('delmas_weather_raw.json', 'delmas')
    #get_historical_weather_data("F:/Data2/job_cedara_debug/dataset_gain_7day/activity_delmas_70101200027_dbft_7_1min.csv", farm_id="cedara_70091100056", out_file="cedara_weather_raw.json", city="Cedara")
    #get_humidity_date('delmas_weather_raw.json', 'delmas')
