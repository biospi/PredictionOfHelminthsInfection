import sys
import tables
from datetime import datetime
import requests
from time import sleep
import json
import os
import pymysql


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def connect_to_sql_database(db_server_name="localhost", db_user="axel", db_password="Mojjo@2015", db_name="south_africa_test4",
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


def get_historical_weather_data():
    farm_id = "Delmas_70101200027"
    days = execute_sql_query("SELECT DISTINCT timestamp_s FROM %s_resolution_d" % farm_id)
    days_ = []
    for item in days:
        days_.append(item['timestamp_s'].split(' ')[0])
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
    URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    purge_file('weather_raw.json')
    with open('weather_raw.json', 'a') as outfile:
        for i, date in enumerate(days_):
            PARAMS = {'key': "b8eced29b72d43129ba154351192201", 'q': "Delmas,south+africa",
                      'date': date, 'tp': 1, 'format': 'json'}
            r = requests.get(url=URL, params=PARAMS)
            data = r.json()
            print(i, len(days_), data)
            json.dump(data, outfile)
            outfile.write('\n')
            sleep(0.5)


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


if __name__ == '__main__':
    print(sys.argv)
    connect_to_sql_database()
    #get_historical_weather_data()
    get_humidity_date('weather_raw.json', 'Delmas')
