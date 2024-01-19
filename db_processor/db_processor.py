import math
import os
import os.path
import re
import statistics
import uuid
import math
from datetime import datetime, timedelta, time
from sys import exit
import itertools

import numpy as np

# import openpyxl
# import tables
# from cassandra.cluster import Cluster
# from ipython_genutils.py3compat import xrange
# from tables import *
import os.path
from collections import defaultdict
import dateutil.relativedelta
import time
import os
import glob

import typer
import xlrd
import pandas as pd
import sys
import pymysql
import tables
import xlrd
from ipython_genutils.py3compat import xrange
from tables import *
from functools import partial
from multiprocessing import Pool

# from openpyxl import load_workbook
# from pycel import ExcelCompiler
# import cryptography #need to be imported or pip install cryptography
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import plotly.express as px
from classifier.src.herd_map import create_herd_map

sql_db = None
MAX_ACTIVITY_COUNT_BIO = 480
pd.set_option("display.max_columns", 48)
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 1000)


# pd.set_option('display.float_format', lambda x: '%d' % x )


class Animal(IsDescription):
    timestamp = Int32Col()
    control_station = Int64Col()
    serial_number = Int64Col()
    signal_strength = Int16Col()
    battery_voltage = Int16Col()
    first_sensor_value = Int32Col()


class Animal2(IsDescription):
    timestamp = Int32Col()
    serial_number = Int64Col()
    signal_strength_min = Int16Col()
    signal_strength_max = Int16Col()
    battery_voltage = Int16Col()
    first_sensor_value = Int32Col()


def execute_sql_query(query, records=None, log_enabled=True):
    try:
        global sql_db
        cursor = sql_db.cursor()
        if records is not None:
            if log_enabled:
                print("SQL Query: %s" % query, records[0])
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


def sql_db_flush():
    global sql_db
    sql_db.commit()


def show_all_records_in_sql_table(table_name):
    execute_sql_query("SELECT * FROM `%s`" % table_name)


def insert_m_record_to_sql_table(table_id, records):
    print("inserting %d records to table %s" % (len(records), table_id))
    query = (
        "INSERT INTO `"
        + table_id
        + "` (timestamp, timestamp_s, serial_number, signal_strength, battery_voltage, "
        "first_sensor_value) VALUES (%s, %s, %s, %s, %s, %s)"
    )
    execute_sql_query(query, records)


def insert_m_record_to_sql_table_(table_id, records):
    print("inserting %d records to table %s" % (len(records), table_id))
    query = (
        "INSERT INTO `"
        + table_id
        + "` (timestamp, timestamp_s, serial_number, signal_strength_max, "
        "signal_strength_min, battery_voltage, first_sensor_value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    execute_sql_query(query, records)


def insert_record_to_sql_table(
    table_id,
    timestamp,
    timestamp_s,
    serial_number_s,
    signal_strength_s,
    battery_voltage_s,
    first_sensor_value,
):
    values = (
        timestamp,
        timestamp_s,
        serial_number_s,
        signal_strength_s,
        battery_voltage_s,
        first_sensor_value,
    )
    query = (
        "INSERT INTO `"
        + table_id
        + "` (timestamp, timestamp_s, serial_number, signal_strength, battery_voltage, "
        "first_sensor_value) VALUES (%d, %s, %d, %d, %d, %d)" % values
    )
    execute_sql_query(query)


def insert_record_to_sql_table_(
    table_id,
    timestamp,
    timestamp_s,
    serial_number,
    signal_strength_max,
    signal_strength_min,
    battery_voltage,
    activity_level_avg,
):
    values = (
        timestamp,
        timestamp_s,
        serial_number,
        signal_strength_max,
        signal_strength_min,
        battery_voltage,
        activity_level_avg,
    )
    query = (
        "INSERT INTO `"
        + table_id
        + "` (timestamp, timestamp_s, serial_number, signal_strength_max,"
        " signal_strength_min, battery_voltage, first_sensor_value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        % values
    )
    execute_sql_query(query)


def drop_all_tables(db_name):
    print("drop all tables in db...")
    tables = execute_sql_query("SHOW TABLES")
    for table in tables:
        name = table["Tables_in_%s" % db_name]
        execute_sql_query("DROP TABLE `%s`" % name)


def create_sql_table(name):
    print("creating sql table %s" % name)
    execute_sql_query(
        "CREATE TABLE `%s` ("
        "id INT AUTO_INCREMENT PRIMARY KEY,"
        "timestamp INT,"
        "timestamp_s VARCHAR(255),"
        "serial_number BIGINT,"
        "signal_strength INT,"
        "battery_voltage INT,"
        "first_sensor_value BIGINT"
        ")" % name
    )


def create_sql_table_(name):
    print("creating sql table %s" % name)
    execute_sql_query(
        "CREATE TABLE `%s` ("
        "id INT AUTO_INCREMENT PRIMARY KEY,"
        "timestamp INT,"
        "timestamp_s VARCHAR(255),"
        "serial_number BIGINT,"
        "signal_strength_min INT,"
        "signal_strength_max INT,"
        "battery_voltage INT,"
        "first_sensor_value BIGINT"
        ")" % name
    )


def create_and_connect_to_sql_db(db_name):
    print("CREATE DATABASE %s..." % db_name)
    # Create a connection object
    db_server_name = "localhost"
    db_user = "root"
    db_password = "Mojjo@2015"
    char_set = "utf8mb4"
    cusror_type = pymysql.cursors.DictCursor
    global sql_db
    sql_db = pymysql.connect(host=db_server_name, user=db_user, password=db_password)
    execute_sql_query("CREATE DATABASE IF NOT EXISTS %s" % db_name)
    connect_to_sql_database(
        db_server_name, db_user, db_password, db_name, char_set, cusror_type
    )


def connect_to_sql_database(
    db_server_name, db_user, db_password, db_name, char_set, cusror_type
):
    print("connecting to db %s..." % db_name)
    global sql_db
    sql_db = pymysql.connect(
        host=db_server_name,
        user=db_user,
        password=db_password,
        db=db_name,
        charset=char_set,
        cursorclass=cusror_type,
    )


def by_size(words, size):
    return [word for word in words if len(word) == size]


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except (FileNotFoundError, PermissionError):
        print("file not found.")


def get_epoch_from_datetime(date, time):
    return int(
        datetime.strptime((date + " " + time), "%d/%m/%y %I:%M:%S %p").timestamp()
    )


def add_record_to_table(table, data):
    sn = data[0]["serial_number"]
    table_row = table.row
    for record in data:

        timestamp = get_epoch_from_datetime(record["date"], record["time"])
        sn = int(sn)
        signal_strength = -1
        battery_voltage = -1
        if (
            record["signal_strength"] is not None
            and re.sub("[^0-9]", "", record["signal_strength"]) != ""
        ):
            signal_strength = -int(re.sub("[^0-9]", "", record["signal_strength"]))
        if (
            record["battery_voltage"] is not None
            and re.sub("[^0-9]", "", record["battery_voltage"]) != ""
        ):
            battery_voltage = int(re.sub("[^0-9]", "", record["battery_voltage"]))
        else:
            try:
                # value is sometimes strored in hex
                battery_voltage = int(record["battery_voltage"], 16)
            except (ValueError, TypeError) as ex:
                print(ex)

        first_sensor_value = int(record["first_sensor_value"])
        x_min, x_max, y_min, y_max, z_min, z_max = 0, 0, 0, 0, 0, 0
        ssv = ""
        if (
            "second_sensor_values_xyz" in record
            and record["second_sensor_values_xyz"] is not None
        ):
            ssv = str(record["second_sensor_values_xyz"])
            split = ssv.split(":")
            # print(split)
            if len(split) == 6:
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    int(split[0]),
                    int(split[1]),
                    int(split[2]),
                    int(split[3]),
                    int(split[4]),
                    int(split[5]),
                )
            # print(x_min, x_max, y_min, y_max, z_min, z_max)
        table_row["timestamp"] = timestamp
        table_row["serial_number"] = int(sn)
        table_row["signal_strength"] = signal_strength
        table_row["battery_voltage"] = battery_voltage
        table_row["first_sensor_value"] = first_sensor_value
        table_row.append()


def add_record_to_table_sum(
    table,
    timestamp_f,
    serial_number_f,
    signal_strenght_max,
    signal_strenght_min,
    battery_voltage_min,
    activity_level_avg,
):
    # print(timestamp_f, serial_number_f, signal_strenght_max, signal_strenght_min, battery_voltage_min, activity_level_avg)
    table_row = table.row
    table_row["timestamp"] = int(timestamp_f)
    table_row["serial_number"] = int(serial_number_f)
    table_row["signal_strength_min"] = signal_strenght_min
    table_row["signal_strength_max"] = signal_strenght_max
    table_row["battery_voltage"] = battery_voltage_min
    table_row["first_sensor_value"] = activity_level_avg
    table_row.append()


def add_record_to_table_single(
    table,
    timestamp_s,
    serial_number_s,
    signal_strength_s,
    battery_voltage_s,
    activity_level_s,
):
    table_row = table.row
    table_row["timestamp"] = timestamp_s
    table_row["serial_number"] = serial_number_s
    table_row["signal_strength"] = signal_strength_s
    table_row["battery_voltage"] = battery_voltage_s
    table_row["first_sensor_value"] = activity_level_s
    # print(timestamp_s, serial_number_s, signal_strenght_s, battery_voltage_s, activity_level_s, table.size_in_memory)
    table_row.append()


def is_same_day(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    return dt1.day == dt2.day


def is_same_month(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    return dt1.month == dt2.month


def is_same_hour(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    return dt1.hour == dt2.hour


def get_elapsed_days(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return rd.days


def get_elapsed_hours(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return rd.hours


def get_elapsed_minutes(dt_1, dt_2):
    return math.ceil((dt_2 - dt_1).total_seconds() / 60)


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return "%02d:%02d:%02d:%02d" % (rd.days, rd.hours, rd.minutes, rd.seconds)


def get_elapsed_time(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return [rd.days, rd.hours, rd.minutes, rd.seconds]


def check_if_same_day(curr_date_time, next_date_time):
    return (
        curr_date_time.year == next_date_time.year
        and curr_date_time.month == next_date_time.month
        and curr_date_time.day == next_date_time.day
        and curr_date_time.hour == next_date_time.hour
        and curr_date_time.minute == next_date_time.minute
    )


def get_first_last_timestamp(animal_records):
    animal_records = sorted(
        animal_records, key=lambda x: x[0]
    )  # make sure that the records are sorted by timestamp
    first_record_date = animal_records[0][0]
    last_record_date = animal_records[-1][0]
    first_record_date_reduced_to_first_sec = datetime.strptime(
        datetime.fromtimestamp(first_record_date).strftime("%Y-%m-%d %H:%M:00"),
        "%Y-%m-%d %H:%M:%S",
    )

    # reduce minutes ex 37 -> 30 or 01 -> 00
    minutes = int(
        str(first_record_date_reduced_to_first_sec.minute).zfill(2)[:-1] + "0"
    )
    first_record_date_reduced_to_first_sec = (
        first_record_date_reduced_to_first_sec.replace(minute=minutes)
    )

    last_record_date_reduced_to_first_sec = datetime.strptime(
        datetime.fromtimestamp(last_record_date).strftime("%Y-%m-%d %H:%M:00"),
        "%Y-%m-%d %H:%M:%S",
    )
    return first_record_date_reduced_to_first_sec, last_record_date_reduced_to_first_sec


def get_list_average(l):
    return int(sum(l) / float(len(l)))


def format_farm_id(farm_id):
    if farm_id == "70091100056":
        farm_id = "cedara_" + farm_id

    if farm_id == "70091100060":
        farm_id = "bothaville_" + farm_id

    if farm_id == "70101100005":
        farm_id = "elandsberg_" + farm_id

    if farm_id == "70101100025":
        farm_id = "eenzaamheid_" + farm_id

    if farm_id == "70101100029":
        farm_id = "msinga_" + farm_id

    if farm_id == "70101200027":
        farm_id = "delmas_" + farm_id

    return farm_id


def init_database(farm_id, database_type="h5"):
    print(sys.argv)

    if database_type == "sql":
        print("store data in sql database...")
        create_sql_table("%s_resolution_min" % farm_id)
        create_sql_table_("%s_resolution_5min" % farm_id)
        create_sql_table_("%s_resolution_10min" % farm_id)
        create_sql_table_("%s_resolution_month" % farm_id)
        create_sql_table_("%s_resolution_week" % farm_id)
        create_sql_table_("%s_resolution_day" % farm_id)
        create_sql_table_("%s_resolution_hour" % farm_id)
        return None, None, None, None, None, None, None

    if database_type == "h5":
        print("store data in h5 database...")
        # init new .h5 file for receiving sorted data
        FILTERS = tables.Filters(complib="blosc", complevel=9)
        compression = False
        if compression:
            # purge_file(farm_id + "_data_compressed_blosc.h5")
            h5file = tables.open_file(
                farm_id + "_data_compressed_blosc.h5",
                "w",
                driver="H5FD_CORE",
                filters=FILTERS,
            )
        else:
            # purge_file(farm_id + ".h5")
            h5file = tables.open_file(farm_id + ".h5", "w", driver="H5FD_CORE")

        group_min = h5file.create_group("/", "resolution_min", "resolution per min")
        group_5min = h5file.create_group("/", "resolution_5min", "resolution per 5min")
        group_10min = h5file.create_group(
            "/", "resolution_10min", "resolution per 10min"
        )
        group_m = h5file.create_group("/", "resolution_month", "resolution per month")
        group_w = h5file.create_group("/", "resolution_week", "resolution per week")
        group_d = h5file.create_group("/", "resolution_day", "resolution per day")
        group_h = h5file.create_group("/", "resolution_hour", "resolution per hour")

        table_min = h5file.create_table(
            group_min, "data", Animal, "Animal data activity level averaged by min"
        )
        table_5min = h5file.create_table(
            group_5min, "data", Animal, "Animal data activity level averaged by 5min"
        )
        table_10min = h5file.create_table(
            group_10min, "data", Animal, "Animal data activity level averaged by 10min"
        )
        table_h = h5file.create_table(
            group_h, "data", Animal2, "Animal data activity level averaged by hour"
        )
        table_d = h5file.create_table(
            group_d, "data", Animal2, "Animal data activity level averaged by day"
        )
        table_w = h5file.create_table(
            group_w, "data", Animal2, "Animal data activity level averaged by week"
        )
        table_m = h5file.create_table(
            group_m, "data", Animal2, "Animal data activity level averaged by month"
        )

        return table_min, table_5min, table_10min, table_h, table_d, table_w, table_m


def resample_to_min(first_timestamp, last_timestamp, animal_records):
    data = []
    n_minutes_in_between = get_elapsed_minutes(first_timestamp, last_timestamp)
    structure_m = {}
    for i in xrange(0, n_minutes_in_between):
        next_timestamp = first_timestamp + timedelta(minutes=i)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT%H:%M")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT%H:%M"
        )
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same hour

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    get_list_average(record[3]),
                    get_list_average(record[4]),
                    activity,
                )
            )

    return data


def custom_round(x, base=5):
    return base * math.floor(x / base)


def resample_to_5min(first_timestamp, last_timestamp, animal_records):
    data = []
    n_minutes_in_between = int(get_elapsed_minutes(first_timestamp, last_timestamp) / 5)
    structure_m = {}
    for i in xrange(0, n_minutes_in_between):
        next_timestamp = first_timestamp + timedelta(minutes=i * 5)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT%H:%M")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT%H:%M"
        )
        minutes_s = str(custom_round(int(record_timestamp_f[-2:]), base=5)).zfill(2)
        record_timestamp_f = record_timestamp_f[:-2] + minutes_s
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same 10min

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def resample_to_10min(first_timestamp, last_timestamp, animal_records):
    data = []
    n_minutes_in_between = int(
        get_elapsed_minutes(first_timestamp, last_timestamp) / 10
    )
    structure_m = {}
    for i in xrange(0, n_minutes_in_between):
        next_timestamp = first_timestamp + timedelta(minutes=i * 10)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT%H:%M")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]
    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT%H:%M"
        )
        minutes_s = str(custom_round(int(record_timestamp_f[-2:]), base=10)).zfill(2)
        record_timestamp_f = record_timestamp_f[:-2] + minutes_s

        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same 10min

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def resample_to_hour(first_timestamp, last_timestamp, animal_records):
    data = []
    n_hours_in_between = int(get_elapsed_minutes(first_timestamp, last_timestamp) / 60)
    structure_m = {}
    for i in xrange(0, n_hours_in_between):
        next_timestamp = first_timestamp + timedelta(hours=i)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT%H:00")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT%H:00"
        )
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same hour

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def resample_to_day(first_timestamp, last_timestamp, animal_records):
    data = []
    n_days_in_between = int(
        get_elapsed_minutes(first_timestamp, last_timestamp) / 60 / 24
    )
    structure_m = {}
    # build the time structure
    for i in xrange(0, n_days_in_between):
        next_timestamp = first_timestamp + timedelta(days=i)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT00:00")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT00:00"
        )
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same hour

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def resample_to_week(first_timestamp, last_timestamp, animal_records):
    data = []
    n_weeks_in_between = int(
        math.ceil(get_elapsed_minutes(first_timestamp, last_timestamp) / 60 / 24 / 7)
    )
    structure_m = {}
    # build the time structure
    for i in xrange(0, n_weeks_in_between):
        next_timestamp = first_timestamp + timedelta(days=i * 7)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT00:00")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT00:00"
        )
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same hour

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def resample_to_month(first_timestamp, last_timestamp, animal_records):
    data = []
    n_months_in_between = int(
        math.ceil(
            get_elapsed_minutes(first_timestamp, last_timestamp) / 60 / 24 / 7 / 30
        )
    )
    structure_m = {}
    # build the time structure
    for i in xrange(0, n_months_in_between):
        next_timestamp = first_timestamp + timedelta(days=i * 30)
        next_timestamp_human_readable = next_timestamp.strftime("%Y-%m-%dT00:00")
        serial_number = animal_records[0][
            2
        ]  # just take first record and get serial number
        structure_m[next_timestamp_human_readable] = [
            int(time.mktime(next_timestamp.timetuple())),
            next_timestamp_human_readable,
            serial_number,
            None,
            None,
            None,
            None,
        ]

    # fill in resampled structure with records data
    for record in animal_records:
        record_timestamp_f = datetime.fromtimestamp(record[0]).strftime(
            "%Y-%m-%dT00:00"
        )
        if record_timestamp_f in structure_m:
            struct = structure_m[record_timestamp_f]
            if struct[3] is None:
                struct[3] = []
            struct[3].append(record[3])  # signal strenght
            if struct[4] is None:
                struct[4] = []
            struct[4].append(record[4])  # battery level
            if struct[5] is None:
                struct[5] = 0
            if record[5] is not None:
                struct[5] += record[5]  # accumulate activity level if in same hour

    # filled in record data in struct save to array for save in db
    for record in structure_m.values():
        activity = record[5]
        if activity is None:
            data.append(record)
        else:
            data.append(
                (
                    record[0],
                    record[1],
                    record[2],
                    max(record[3]),
                    min(record[3]),
                    min(record[4]),
                    activity,
                )
            )
    return data


def process_raw_h5files(path, njob=None):
    print(path)
    h5_raw = tables.open_file(path, "r")
    data = h5_raw.root.table
    list_raw = []
    print("loading data...")
    cpt_total = 0
    cpt_negative = 0
    cpt_above_bio = 0
    all_data_point = []
    for idx, x in enumerate(data):  # need idx for data iteration?
        farm_id = x["control_station"]
        if "cedara" in str(path) and farm_id != 70091100056:  # todo remove
            continue

        if farm_id != 70101200027:
            continue

        if datetime.fromtimestamp(x["timestamp"]).year < 2016:
            continue
        # if x['serial_number'] not in [40121100797]:
        #     continue
        # if len(str(x['serial_number'])) != len("70091100056"): #todo remove
        #     continue
        if x['first_sensor_value'] < 0:
            cpt_negative += 1
        if x['first_sensor_value'] > MAX_ACTIVITY_COUNT_BIO:
            cpt_above_bio += 1

        all_data_point.append(x['first_sensor_value'])
        if x['first_sensor_value'] > MAX_ACTIVITY_COUNT_BIO or x['first_sensor_value'] < 0:
            continue

        value = (
            x["timestamp"],
            farm_id,
            x["serial_number"],
            x["signal_strength"],
            x["battery_voltage"],
            x["first_sensor_value"],
            datetime.fromtimestamp(x["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S"),
            datetime.strptime(
                datetime.fromtimestamp(x["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S"),
                "%Y-%m-%dT%H:%M:%S",
            ),
        )
        list_raw.append(value)
        # if idx > 100000:  # todo remove
        #     break
    print(f"cpt_negative={cpt_negative} cpt_above_bio={cpt_above_bio}")
    # records = np.array(all_data_point)
    # df = pd.DataFrame()
    # df["counts"] = records
    # df = df[df["counts"] > 0]
    # df = df[df["counts"] < np.exp(10)]
    # df["activity counts (logged)"] = np.log(df["counts"])
    # df = df.astype(float)
    # fig = px.histogram(df, x="activity counts (logged)", title="Histogram of positive raw activity counts", nbins=1000)
    # fig.add_vline(x=np.log(MAX_ACTIVITY_COUNT_BIO), line_dash='dash', line_color='firebrick')
    # #fig.show()
    # filepath = f"{Path(path).stem}.html"
    # print(filepath)
    # fig.write_html(filepath)

    #group records by farm id/control_station
    groups = defaultdict(list)
    for i, obj in enumerate(list_raw):
        groups[obj[1]].append(obj)
    animal_list_grouped_by_farmid = list(groups.values())

    for group in animal_list_grouped_by_farmid:
        farm_id = str(group[0][1])
        process_raw_file(farm_id, group, njob=njob)


def create_mean_median_animal_(data):
    print("create_mean_median_animal...")
    # group records by epoch
    groups = defaultdict(list)
    for i, obj in enumerate(data):
        groups[obj[1]].append(obj)
    grouped_by_epoch = list(groups.values())
    mean_data = []
    median_data = []
    for item in grouped_by_epoch:
        mean_activity = {}
        mean_battery = 0
        mean_signal_strengh = 0
        epoch = item[0][0]
        epoch_f = item[0][1]
        mean_serial_number = 50000000000
        median_serial_number = 60000000000
        median_activity = {}
        median_battery = {}
        median_signal_strengh = {}
        for record in item:
            # print(record)
            median_activity[record[2]] = record[5] if record[5] is not None else 0
            median_battery[record[2]] = record[4] if record[4] is not None else 0
            median_signal_strengh[record[2]] = record[3] if record[3] is not None else 0

            if record[1] not in mean_activity:
                mean_activity[record[1]] = []

            if record[5] is not None and record[5] > 0:
                mean_activity[record[1]].append(record[5])

            mean_battery += record[4] if record[4] is not None else 0
            mean_signal_strengh += record[3] if record[3] is not None else 0

        ss = int(mean_signal_strengh / len(item))
        bat = int(mean_battery / len(item))
        m_a_l = list(mean_activity.values())[0]
        m_a = None if len(m_a_l) == 0 else sum(m_a_l) / len(m_a_l)
        mean_record = (
            epoch,
            epoch_f,
            mean_serial_number,
            ss if ss != 0 else None,
            bat if bat != 0 else None,
            m_a,
        )
        mean_data.append(mean_record)

        median_a = statistics.median(sorted(list(median_activity.values())))
        median_b = statistics.median(sorted(list(median_battery.values())))
        median_ss = statistics.median(sorted(list(median_signal_strengh.values())))
        median_record = (
            epoch,
            epoch_f,
            median_serial_number,
            median_ss if median_ss != 0 else None,
            median_b if median_b != 0 else None,
            median_a if median_b != 0 else None,
        )
        median_data.append(median_record)
    result = mean_data + median_data
    return result


def create_indexes(farm_id):
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_month__sn__ts on %s_resolution_month(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_week__sn__ts on %s_resolution_week(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_day__sn__ts on %s_resolution_day(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_hour__sn__ts on %s_resolution_hour(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_10min__sn__ts on %s_resolution_10min(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_5min__sn__ts on %s_resolution_5min(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )
    execute_sql_query(
        "CREATE INDEX ix__%s_resolution_min__sn__ts on %s_resolution_min(serial_number, timestamp, first_sensor_value )"
        % (farm_id, farm_id),
        log_enabled=True,
    )


def create_mean_median_animal(data):
    print("create_mean_median_animal...")
    # group records by epoch
    groups = defaultdict(list)
    for i, obj in enumerate(data):
        groups[obj[1]].append(obj)
    grouped_by_epoch = list(groups.values())
    mean_data = []
    median_data = []

    for item in grouped_by_epoch:
        mean_activity = {}
        mean_battery = 0
        mean_max_signal_strengh = 0
        mean_min_signal_strengh = 0
        epoch = item[0][0]
        epoch_f = item[0][1]
        mean_serial_number = 50000000000
        median_serial_number = 60000000000
        median_activity = {}
        median_battery = {}
        median_max_signal_strengh = {}
        median_min_signal_strengh = {}
        for record in item:
            median_activity[record[2]] = record[6] if record[6] is not None else 0
            median_battery[record[2]] = record[5] if record[5] is not None else 0
            median_max_signal_strengh[record[2]] = (
                record[4] if record[4] is not None else 0
            )
            median_min_signal_strengh[record[2]] = (
                record[3] if record[3] is not None else 0
            )

            if record[1] not in mean_activity:
                mean_activity[record[1]] = []

            if record[6] is not None and record[6] > 0:
                mean_activity[record[1]].append(record[6])

            mean_battery += record[5] if record[5] is not None else 0
            mean_max_signal_strengh += record[4] if record[4] is not None else 0
            mean_min_signal_strengh += record[3] if record[3] is not None else 0

        ss_min = int(mean_min_signal_strengh / len(item))
        ss_max = int(mean_max_signal_strengh / len(item))
        bat = int(mean_battery / len(item))
        m_a_l = list(mean_activity.values())[0]
        m_a = None if len(m_a_l) == 0 else sum(m_a_l) / len(m_a_l)
        mean_record = (
            epoch,
            epoch_f,
            mean_serial_number,
            ss_min if ss_min != 0 else None,
            ss_max if ss_max != 0 else None,
            bat if bat != 0 else None,
            m_a,
        )

        mean_data.append(mean_record)

        median_a = statistics.median(sorted(list(median_activity.values())))
        median_b = statistics.median(sorted(list(median_battery.values())))
        median_min_ss = statistics.median(
            sorted(list(median_min_signal_strengh.values()))
        )
        median_max_ss = statistics.median(
            sorted(list(median_max_signal_strengh.values()))
        )
        median_record = (
            epoch,
            epoch_f,
            median_serial_number,
            median_min_ss if median_min_ss != 0 else None,
            median_max_ss if median_max_ss != 0 else None,
            median_b if median_b != 0 else None,
            median_a if median_b != 0 else None,
        )
        median_data.append(median_record)

    result = mean_data + median_data
    print("median animal lenght=", len(median_data))
    print("mean animal lenght=", len(mean_data))

    return result


def create_dataframe(list_data, res):
    df = pd.DataFrame(list_data)
    df.columns = [
        "timestamp",
        "farm_id",
        "serial_number",
        "signal_strength",
        "battery_voltage",
        "first_sensor_value",
        "date",
        "date_str",
    ]
    df = df.drop_duplicates(subset=["timestamp", "first_sensor_value"])
    df = df.sort_values(by="date")
    if res is not None:
        df.rename({"date": "index"}, axis=1, inplace=True)
    return df


def using_clump(a):
    return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def isNaN(num):
    return num != num


def thresholded_interpol(df_1min, thresh):
    # print("thresholded_interpol...", thresh)
    data = pd.DataFrame(df_1min["first_sensor_value"])
    mask = data.copy()
    df = pd.DataFrame(data["first_sensor_value"])
    df["new"] = (df.notnull() != df.shift().notnull()).cumsum()
    df["ones"] = 1
    mask["first_sensor_value"] = (
        df.groupby("new")["ones"].transform("count") < thresh
    ) | data["first_sensor_value"].notnull()

    interpolated = data.interpolate().bfill()[mask]
    if thresh > 0:
        df_1min["first_sensor_value"] = interpolated
    else:
        df_1min["first_sensor_value"] = data

    # df_1min["first_sensor_value"] = interpolated
    # histogram_array_nan_dur, histogram_array_no_activity_dur = [], []
    # clump = using_clump(data["first_sensor_value"].values)
    len_holes = [
        len(list(g))
        for k, g in itertools.groupby(
            data["first_sensor_value"].values, lambda x: np.isnan(x)
        )
        if k
    ]
    # for nan_gap in len_holes:
    #     histogram_array_nan_dur.append(nan_gap)

    len_no_activity = [
        len(list(g))
        for k, g in itertools.groupby(
            data["first_sensor_value"].values, lambda x: x == 0
        )
        if k
    ]
    # for zero_gap in len_no_activity:
    #     histogram_array_no_activity_dur.append(zero_gap)
    df_linterpolated = df_1min.copy()
    df_linterpolated["signal_strength_2"] = df_linterpolated["signal_strength"]
    df_linterpolated["timestamp"] = (
        df_linterpolated.index.values.astype(np.int64) // 10 ** 9
    )
    df_linterpolated["date_str"] = df_linterpolated.index.strftime("%Y-%m-%dT%H:%M")
    df_linterpolated = df_linterpolated.assign(
        farm_id=df_linterpolated["farm_id"].max()
    )

    df_linterpolated = df_linterpolated[
        [
            "timestamp",
            "date_str",
            "serial_number",
            "signal_strength",
            "signal_strength_2",
            "battery_voltage",
            "first_sensor_value",
        ]
    ]

    df_linterpolated = df_linterpolated.assign(
        serial_number=df_linterpolated["serial_number"].max()
    )  # fills in gap when agg fails because of empty sensor value

    df_linterpolated[
        ["signal_strength", "signal_strength_2", "battery_voltage"]
    ] = df_linterpolated[
        ["signal_strength", "signal_strength_2", "battery_voltage"]
    ].fillna(
        value=0
    )

    print("thresholded_interpol done.", thresh)
    return df_linterpolated


def resample_df(res, data):
    # print("resample_df ", res)
    # data = data.set_index('index')
    # data.index.name = None
    data.index = pd.to_datetime(data.index)
    df = data.resample(res).agg(
        dict(
            timestamp="first",
            serial_number="first",
            signal_strength="min",
            signal_strength_2="max",
            battery_voltage="mean",
            first_sensor_value="sum",
            date_str="first",
        ),
        skipna=False,
    )
    df = df[
        [
            "timestamp",
            "date_str",
            "serial_number",
            "signal_strength",
            "signal_strength_2",
            "battery_voltage",
            "first_sensor_value",
        ]
    ]
    # df['timestamp'] = df.index.values.astype(np.int64) // 10 ** 9
    # df['date_str'] = df.index.strftime('%Y-%m-%dT%H:%M')
    # df.loc[np.isnan(df['timestamp']), 'first_sensor_value'] = np.nan
    # df['date_str'] = df.index.strftime('%Y-%m-%dT%H:%M')
    # df['timestamp'] = df.index.values.astype(np.int64) // 10 ** 9
    # df = df.reset_index()
    # df.loc[df.isnull().any(axis=1), ["first_sensor_value"]] = np.nan
    return df


def resample_1min_simple(data):
    print("resample_1min_simple...")
    # data = data.set_index('index')
    # data.index.name = None
    data.index = pd.to_datetime(data["index"])
    data.index.name = None
    del data["index"]
    df = data.resample("1min").agg(
        dict(
            timestamp="first",
            farm_id="first",
            serial_number="first",
            signal_strength="min",
            battery_voltage="mean",
            first_sensor_value="sum",
            date_str="first",
        ),
        skipna=False,
    )
    df = df[
        [
            "timestamp",
            "date_str",
            "farm_id",
            "serial_number",
            "signal_strength",
            "battery_voltage",
            "first_sensor_value",
        ]
    ]
    # df['timestamp'] = df.index.values.astype(np.int64) // 10 ** 9
    # df['date_str'] = df.index.strftime('%Y-%m-%dT%H:%M')
    df.loc[np.isnan(df["timestamp"]), "first_sensor_value"] = np.nan
    df["date_str"] = df.index.strftime("%Y-%m-%dT%H:%M")
    df["timestamp"] = df.index.values.astype(np.int64) // 10 ** 9
    df["serial_number"] = df["serial_number"][df["serial_number"].notnull()].values[0]
    df["farm_id"] = df["farm_id"][df["farm_id"].notnull()].values[0]
    # df = df.reset_index()
    # df.loc[df.isnull().any(axis=1), ["first_sensor_value"]] = np.nan
    return df


def resample_1min(df_raw):
    start = time.time()
    print("resample_1min...")
    df_raw = df_raw.set_index("index")
    df_raw.index.name = None
    df_raw.index = pd.to_datetime(df_raw.index)
    start_date = min(df_raw.index.array).replace(second=0)
    end_date = max(df_raw.index.array).replace(second=0)
    date_range = np.array(
        pd.date_range(start_date, end_date, freq="1T", normalize=False)
    )
    df_1min = pd.DataFrame(date_range)
    df_1min.index = date_range
    df_raw.index.name = "index"
    for col in df_raw.columns:
        df_1min[col] = np.nan
    del df_1min[0]

    t0 = df_1min.index[0].value
    dt = df_1min.index[1].value - df_1min.index[0].value

    N = df_raw.shape[0] - 1
    i = N
    while i > 0:
        row_raw = df_raw.iloc[i, :]
        x = df_raw.index[i].value
        bin = int((x - t0) / dt)
        if isNaN(df_1min.at[df_1min.index[bin], "timestamp"]):
            df_1min.at[df_1min.index[bin], "timestamp"] = row_raw.timestamp
            df_1min.at[df_1min.index[bin], "farm_id"] = row_raw.farm_id
            df_1min.at[df_1min.index[bin], "serial_number"] = row_raw.serial_number
            df_1min.at[df_1min.index[bin], "signal_strength"] = row_raw.signal_strength
            df_1min.at[df_1min.index[bin], "battery_voltage"] = row_raw.battery_voltage
            df_1min.at[
                df_1min.index[bin], "first_sensor_value"
            ] = row_raw.first_sensor_value
            df_1min.at[df_1min.index[bin], "date_str"] = row_raw.date_str
        else:
            # print("Multiple binned time stamps, preforming a shift:")
            repeat = 1
            space = 0
            begIdx = i
            j = begIdx - 1
            endIdx = 0
            endBin = 0
            x0_bin = bin
            #            fail = 0
            while True:
                # For debuging
                # print("idx x0: %d  idx x1: %d \n" % (j+1, j), df_raw.iloc[list(range(j, j + 2)), :])
                x1 = df_raw.index[j].value
                x1_bin = int((x1 - t0) / dt)
                dBin = x0_bin - x1_bin - 1
                if dBin < 0:
                    repeat += 1
                elif dBin > 0:
                    space += dBin
                if space >= repeat:
                    endIdx = j
                    endBin = x1_bin
                    break
                x0_bin = x1_bin
                j -= 1
            #                fail += 1
            #                if fail > 50:
            #                    break
            #            if fail > 50:
            #                print("Error with time stamps in data: !!!")
            #                print("idx x0: %d  idx x1: %d \n" % (begIdx, j), df_raw.iloc[list(range(j, begIdx + 2)), :])
            #                break
            # print("Space found: Realign data to correct time stamp...")
            # print("Raw data Index: [Begin, End] =  [%d, %d]" % (begIdx, endIdx))
            for k in range(begIdx, endIdx - 1, -1):
                # For debuging
                # print("idx x0: %d \n" % k, df_raw.iloc[list(range(k, k +1)), :])
                row_raw = df_raw.iloc[k, :]
                bin -= 1
                if isNaN(df_1min.at[df_1min.index[bin], "timestamp"]):
                    df_1min.at[df_1min.index[bin], "timestamp"] = row_raw.timestamp
                    df_1min.at[df_1min.index[bin], "farm_id"] = row_raw.farm_id
                    df_1min.at[
                        df_1min.index[bin], "serial_number"
                    ] = row_raw.serial_number
                    df_1min.at[
                        df_1min.index[bin], "signal_strength"
                    ] = row_raw.signal_strength
                    df_1min.at[
                        df_1min.index[bin], "battery_voltage"
                    ] = row_raw.battery_voltage
                    df_1min.at[
                        df_1min.index[bin], "first_sensor_value"
                    ] = row_raw.first_sensor_value
                    df_1min.at[df_1min.index[bin], "date_str"] = row_raw.date_str
            i = endIdx
        i -= 1

    # for row_raw in df_raw.itertuples():
    #     x = row_raw.Index.value
    #     bin = int((x-t0)/dt)
    #     if isNaN(df_1min.at[df_1min.index[bin], 'timestamp']):
    #         df_1min.at[df_1min.index[bin], 'timestamp'] = row_raw.timestamp
    #         df_1min.at[df_1min.index[bin], 'farm_id'] = row_raw.farm_id
    #         df_1min.at[df_1min.index[bin], 'serial_number'] = row_raw.serial_number
    #         df_1min.at[df_1min.index[bin], 'signal_strength'] = row_raw.signal_strength
    #         df_1min.at[df_1min.index[bin], 'battery_voltage'] = row_raw.battery_voltage
    #         df_1min.at[df_1min.index[bin], 'first_sensor_value'] = row_raw.first_sensor_value
    #         df_1min.at[df_1min.index[bin], 'date_str'] = row_raw.date_str
    #     else:
    #         if isNaN(df_1min.at[df_1min.index[bin+1], 'timestamp']):
    #             df_1min.at[df_1min.index[bin+1], 'timestamp'] = row_raw.timestamp
    #             df_1min.at[df_1min.index[bin+1], 'farm_id'] = row_raw.farm_id
    #             df_1min.at[df_1min.index[bin+1], 'serial_number'] = row_raw.serial_number
    #             df_1min.at[df_1min.index[bin+1], 'signal_strength'] = row_raw.signal_strength
    #             df_1min.at[df_1min.index[bin+1], 'battery_voltage'] = row_raw.battery_voltage
    #             df_1min.at[df_1min.index[bin+1], 'first_sensor_value'] = row_raw.first_sensor_value
    #             df_1min.at[df_1min.index[bin+1], 'date_str'] = row_raw.date_str
    #         else:
    #             print("multiple bin warning ", row_raw.timestamp, bin, row_raw.date_str, idx)
    #             print(df_raw.iloc[list(range(idx-4, idx+4)), :])
    #     idx += 1
    # i = 0
    # scaning_range = 3
    # for row_1min in df_1min.itertuples():
    #     date_1min = row_1min.Index
    #     if i == df_raw.shape[0] - scaning_range:
    #         break
    #     df_to_scan = df_raw.iloc[list(range(i, i+scaning_range)), :]
    #     cpt_not_same_min = 0
    #     for row_raw in df_to_scan.itertuples():
    #         date_raw = row_raw.Index
    #         if is_in_same_minute(date_1min, date_raw):
    #             cpt_not_same_min += 1
    #             if isNaN(df_1min.at[row_1min.Index, 'timestamp']):
    #                 df_1min.at[row_1min.Index, 'timestamp'] = row_raw.timestamp
    #                 df_1min.at[row_1min.Index, 'farm_id'] = row_raw.farm_id
    #                 df_1min.at[row_1min.Index, 'serial_number'] = row_raw.serial_number
    #                 df_1min.at[row_1min.Index, 'signal_strength'] = row_raw.signal_strength
    #                 df_1min.at[row_1min.Index, 'battery_voltage'] = row_raw.battery_voltage
    #                 df_1min.at[row_1min.Index, 'first_sensor_value'] = row_raw.first_sensor_value
    #                 df_1min.at[row_1min.Index, 'date_str'] = row_raw.date_str
    #             else:
    #                 next_time = row_1min.Index + timedelta(minutes=1)
    #                 df_1min.at[next_time, 'timestamp'] = row_raw.timestamp
    #                 df_1min.at[next_time, 'farm_id'] = row_raw.farm_id
    #                 df_1min.at[next_time, 'serial_number'] = row_raw.serial_number
    #                 df_1min.at[next_time, 'signal_strength'] = row_raw.signal_strength
    #                 df_1min.at[next_time, 'battery_voltage'] = row_raw.battery_voltage
    #                 df_1min.at[next_time, 'first_sensor_value'] = row_raw.first_sensor_value
    #                 df_1min.at[next_time, 'date_str'] = row_raw.date_str
    #     if cpt_not_same_min == 0:
    #         continue
    #     i += 1
    end = time.time()
    print("Elapsed = %s" % (end - start))
    # print(df_raw)
    # print(df_1min)
    return df_1min


def is_in_same_dminute(date1, date2):
    if date1.date() != date2.date():
        return False
    if date1.hour != date2.hour:
        return False
    if date1.minute != date2.minute:
        return False
    return True


def get_db_data(df):
    data = [
        (
            np.nan if np.isnan(x[0]) else int(x[0]),
            str(x[6]),
            None if np.isnan(x[2]) else int(x[2]),
            None if np.isnan(x[3]) else int(x[3]),
            None if np.isnan(x[4]) else int(x[4]),
            None if np.isnan(x[5]) else int(x[5]),
        )
        for x in df.to_numpy()
    ]
    # activity = [None if np.isnan(x[6]) else int(x[6]) for x in subset.to_numpy()]
    return data


def resample(res, data, idx):
    print(f"start resample {idx}")
    df = resample_df(res, data)
    df["timestamp"] = df.index.values.astype(np.int64) // 10 ** 9
    df["date_str"] = df.index.strftime("%Y-%m-%dT%H:%M")

    subset = df[
        [
            "timestamp",
            "date_str",
            "serial_number",
            "signal_strength",
            "signal_strength_2",
            "battery_voltage",
            "first_sensor_value",
        ]
    ]

    subset = subset.assign(
        serial_number=df["serial_number"].max()
    )  # fills in gap when agg fails because of empty sensor value
    # subset = subset.assign(serial_number=df["farm_id"].max())
    subset = subset.assign(signal_strength=df["signal_strength"].max())
    subset = subset.assign(signal_strength_2=df["signal_strength_2"].max())
    subset = subset.assign(battery_voltage=df["battery_voltage"].max())

    data = [
        (
            np.nan if np.isnan(x[0]) else int(x[0]),
            str(x[1]),
            None if np.isnan(x[2]) else int(x[2]),
            None if np.isnan(x[3]) else int(x[3]),
            None if np.isnan(x[4]) else int(x[4]),
            None if np.isnan(x[5]) else int(x[5]),
            None if np.isnan(x[6]) else int(x[6]),
        )
        for x in subset.to_numpy()
    ]

    # activity = [None if np.isnan(x[6]) else int(x[6]) for x in subset.to_numpy()]
    print(f"resample {idx} done.")
    return data


def build_data_from_raw(animal_records_df, threshold_gap, idx, res=None):
    df_raw = resample_1min(animal_records_df)
    df_linterpolated = thresholded_interpol(df_raw, threshold_gap)
    df_processed = resample(res, df_linterpolated, idx)
    return df_processed


def process(animal_records, threshold_gap, res, idx=None, t=None):
    if idx is not None:
        print(f"process progress {idx}/{t}...")
    animal_records_df = create_dataframe(animal_records, res)
    if res is None:
        d = get_db_data(animal_records_df)
        return [d]

    df_ = build_data_from_raw(animal_records_df, threshold_gap, idx, res=res)

    if idx is not None:
        print(f"done {res} {idx}/{t}...")
    return [df_]


async_results_1min = []
async_results_5min = []
async_results_10min = []
async_results_day = []
async_results_hour = []
async_results_week = []
async_results_month = []


def save_result_1min(result):
    async_results_1min.append(result)


def save_result_5min(result):
    async_results_5min.append(result)


def save_result_10min(result):
    async_results_10min.append(result)


def save_result_day(result):
    async_results_day.append(result)


def save_result_hour(result):
    async_results_hour.append(result)


def save_result_week(result):
    async_results_week.append(result)


def save_result_month(result):
    async_results_month.append(result)


def export_data_to_csv(data, farm_id, threshold_gap, resolution):
    print("exporting data...")
    filename = "%s_%d_%s_debug.csv" % (farm_id, threshold_gap, resolution)
    df = pd.DataFrame(data)
    df.columns = ["epoch", "datetime", "animal_id", "s1", "s2", "battery", "activity"]
    df_sub = df[["epoch", "datetime", "animal_id", "activity"]]
    purge_file(filename)
    df_sub.to_csv(filename, sep=",", index=False)
    print(filename)


def export_rawdata_to_csv(df, farm_id):
    print("exporting data...")
    animal_id = int(
        df[df["serial_number"].isna() == False]["serial_number"].unique()[0]
    )
    path = "csv_debug/%s/" % (farm_id)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename_path = path + "%d_raw.csv" % (animal_id)
    df_sub = df[["timestamp", "first_sensor_value"]]
    df_sub["timestamp"] = df_sub.index.values.astype(np.int64) // 10 ** 9
    purge_file(filename_path)
    df_sub.to_csv(filename_path, sep=",", index=False)
    print(filename_path)


def process_raw_file(farm_id, data, threshold_gap=-1, njob=None, database_type="sql", thresh=1000):

    start_time = time.time()
    farm_id = format_farm_id(farm_id)
    table_min, table_5min, table_10min, table_h, table_d, table_w, table_m = init_database(farm_id)
    print("process data for farm %s." % farm_id)

    # group records by animal id/serial number
    groups = defaultdict(list)
    for obj in data:
        groups[obj[2]].append(obj)
    animal_list_grouped_by_serialn = list(groups.values())

    data_resampled_min = []
    data_resampled_5min = []
    data_resampled_10min = []
    data_resampled_hour = []
    data_resampled_day = []
    data_resampled_week = []
    data_resampled_month = []

    MULTI_THREADING_ENABLED = njob > 1

    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=njob)
        for idx, animal_records in enumerate(animal_list_grouped_by_serialn):
            if len(animal_records) <= thresh:
                continue
            print("animal_records=", len(animal_records))
            # pool.apply_async(process, (farm_id, animal_records, zero_to_nan_threh, interpolation_thesh,),
            # callback=save_result_10min)

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    None,
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_1min,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "5min",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_5min,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "10min",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_10min,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "D",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_day,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "H",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_hour,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "W",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_week,
            )

            pool.apply_async(
                process,
                (
                    animal_records,
                    threshold_gap,
                    "M",
                    idx,
                    len(animal_list_grouped_by_serialn),
                ),
                callback=save_result_month,
            )

        pool.close()
        pool.join()
        for item in async_results_1min:
            data_resampled_min.extend(item[0])
        for item in async_results_5min:
            data_resampled_5min.extend(item[0])
        for item in async_results_10min:
            data_resampled_10min.extend(item[0])
        print("saving pool results...")
        for item in async_results_day:
            data_resampled_day.extend(item[0])
        print("saving pool results...")
        for item in async_results_hour:
            data_resampled_hour.extend(item[0])
        print("saving pool results...")
        for item in async_results_week:
            data_resampled_week.extend(item[0])
        print("saving pool results...")
        for item in async_results_month:
            data_resampled_month.extend(item[0])
    else:
        for idx, animal_records in enumerate(animal_list_grouped_by_serialn):
            print("progress=%d/%d." % (idx, len(animal_list_grouped_by_serialn)))
            if len(animal_records) <= thresh:
                continue
            print("animal_records=", len(animal_records))

            result = process(animal_records, threshold_gap, None)
            data_resampled_min.extend(result[0])

            result = process(animal_records, threshold_gap, "5min")
            data_resampled_5min.extend(result[0])

            result = process(animal_records, threshold_gap, "10min")
            data_resampled_10min.extend(result[0])

            result = process(animal_records, threshold_gap, "D")
            data_resampled_day.extend(result[0])

            result = process(animal_records, threshold_gap, "H")
            data_resampled_hour.extend(result[0])

            result = process(animal_records, threshold_gap, "W")
            data_resampled_week.extend(result[0])

            result = process(animal_records, threshold_gap, "M")
            data_resampled_month.extend(result[0])

    # plot_histogram(histogram_array_nan_dur, farm_id, threshold_gap, "histogram of nan gaps")
    #
    # plot_histogram(histogram_array_zero_dur, farm_id, threshold_gap, "histogram of zero gaps")

    histogram_array_nan_dur = []

    if database_type == "sql":
        print("saving data to db...")
        # create_herd_map(farm_id, None, np.array(pd.DataFrame(activity_data).values.tolist()), animals_id, None, fontsize=50)
        # exit(0)

        data_resampled_month.extend(create_mean_median_animal(data_resampled_month))
        insert_m_record_to_sql_table_("%s_resolution_month" % farm_id, data_resampled_month)

        data_resampled_week.extend(create_mean_median_animal(data_resampled_week))
        insert_m_record_to_sql_table_("%s_resolution_week" % farm_id, data_resampled_week)

        data_resampled_day.extend(create_mean_median_animal(data_resampled_day))
        insert_m_record_to_sql_table_("%s_resolution_day" % farm_id, data_resampled_day)

        data_resampled_hour.extend(create_mean_median_animal(data_resampled_hour))
        insert_m_record_to_sql_table_("%s_resolution_hour" % farm_id, data_resampled_hour)

        data_resampled_10min.extend(create_mean_median_animal(data_resampled_10min))
        insert_m_record_to_sql_table_("%s_resolution_10min" % farm_id, data_resampled_10min)

        data_resampled_5min.extend(create_mean_median_animal(data_resampled_5min))
        insert_m_record_to_sql_table_("%s_resolution_5min" % farm_id, data_resampled_5min)

        data_resampled_min.extend(create_mean_median_animal_(data_resampled_min))
        insert_m_record_to_sql_table("%s_resolution_min" % farm_id, data_resampled_min)

        create_indexes(farm_id)
        sql_db_flush()

    if database_type == 'h5':
        table_m.append(data_resampled_month)
        table_w.append(data_resampled_week)
        table_d.append(data_resampled_day)
        table_h.append(data_resampled_hour)
        table_min.append(data_resampled_min)
        table_10min.append(data_resampled_min)
        table_5min.append(data_resampled_min)
        table_m.flush()
        table_w.flush()
        table_d.flush()
        table_h.flush()
        table_min.flush()
        table_10min.flush()
        table_5min.flush()

    print(
        len(data_resampled_5min),
        len(data_resampled_10min),
        len(data_resampled_hour),
        len(data_resampled_day),
        len(data_resampled_week),
        len(data_resampled_month),
    )
    print(get_elapsed_time_string(start_time, time.time()))
    print("finished processing raw file.")


def xl_date_to_date(xldate, wb):
    year, month, day, hour, minute, second = xlrd.xldate_as_tuple(xldate, wb.datemode)
    return "%02d/%02d/%d" % (day, month, year)


def convert_excel_time(cell_with_excel_time, wb):
    year, month, day, hour, minute, second = xlrd.xldate_as_tuple(
        cell_with_excel_time, wb.datemode
    )  # year=month=day=0
    py_date = datetime(
        year=1989, month=4, day=12, hour=hour, minute=minute, second=second
    )  # used dummy date we only need time
    time_value = py_date.strftime("%I:%M:%S %p")
    return time_value


def dump_cell(sheet, rowx, colx):
    c = sheet.cell(rowx, colx)
    xf = sheet.book.xf_list[c.xf_index]
    fmt_obj = sheet.book.format_map[xf.format_key]
    print(
        rowx,
        colx,
        repr(c.value),
        c.ctype,
        fmt_obj.type,
        fmt_obj.format_key,
        fmt_obj.format_str,
    )
    return [
        rowx,
        colx,
        repr(c.value),
        c.ctype,
        fmt_obj.type,
        fmt_obj.format_key,
        fmt_obj.format_str,
    ]


def empty_list(l):
    return len(l) == l.count("")


def print_except(e):
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    print(message)


def strip_list(l):
    # print(l)
    return [
        x.strip().lower().replace(" ", "").replace("batteryife", "batterylife")
        for x in l
    ]


def ignore(path):
    to_ignore = False
    IGNORE_LIST = ["AktsMeting Senst-Cedara Bok 1201024", "Cedara tye dubbeld"]
    for item in IGNORE_LIST:
        if item in path:
            to_ignore = True
            break
    return to_ignore


def generate_raw_files_from_xlsx(directory_path, file_name):
    start_time = time.time()
    print("start readind xls files...")
    purge_file("log.txt")
    log_file = open("log.txt", "a")
    os.chdir(directory_path)
    file_paths = [
        val
        for sublist in [
            [os.path.join(i[0], j) for j in i[2] if j.endswith(".xlsx")]
            for i in os.walk(directory_path)
        ]
        for val in sublist
    ]
    print("founded %d files" % len(file_paths))
    print(file_paths)
    print("start generating raw file...")
    compression = False
    if compression:
        purge_file("raw_data_compressed_blosc_raw.h5")
        h5file = tables.open_file(
            "raw_data_compressed_blosc_raw.h5",
            "w",
            driver="H5FD_CORE",
            filters=tables.Filters(complib="blosc", complevel=9),
        )
    else:
        purge_file(file_name)
        # h5file = tables.open_file("raw_data.h5", mode="w", driver="H5FD_CORE")

        store = pd.HDFStore(file_name)

    # table_f = h5file.create_table("/", "data", Animal, "Animal data in full resolution", expectedrows=33724492)
    # table_row = table_f.row
    valid_rows = 0

    for curr_file, path in enumerate(file_paths):
        print("progress %s/%d" % (curr_file, len(file_paths)))
        if ignore(path):
            continue
        # if "70101200027_2015-4-15_06-00-00_to_2015-4-20_06-00-00.xlsx" not in path:
        #     # print(curr_file)
        #     continue

        # path = 'C:\\SouthAfrica\\Tracking Data\\Delmas\\June 2015\\70101200027_2015-5-31_06-00-00_to_2015-6-03_06-00-00.xlsx'
        # table_f = h5file.create_table("/", "data%d" % curr_file, Animal, path, expectedrows=33724492)
        # table_row = table_f.row
        df = []
        transponders = {}
        try:
            record_log = ""
            print("loading file in memory for reading...")
            print(path)
            book = xlrd.open_workbook(path)
            for sheet in book.sheets():
                print("reading sheet", sheet.name)
                # sheet = book.sheet_by_index(0)
                try:
                    farm_id = int(sheet.name.split("_")[0])
                except ValueError as e:
                    print(e)
                    print(path)

                print("start reading...")
                found_col_index = False
                for row_index in xrange(0, sheet.nrows):
                    # if row_index > 10:
                    #     break
                    try:
                        row_values = [
                            sheet.cell(row_index, col_index).value
                            for col_index in xrange(0, sheet.ncols)
                        ]
                        if empty_list(row_values):
                            continue
                        # print(row_values)

                        if (
                            "Cedara" in directory_path
                            and "Sender" in path
                            and not found_col_index
                        ):
                            date_col_index = 0
                            time_col_index = 1
                            control_station_col_index = 2
                            try:
                                farm_id = int(row_values[control_station_col_index])
                            except ValueError as e:
                                print(e)
                            serial_number_col_index = 3
                            signal_strength_col_index = 4
                            battery_voltage_col_index = 5
                            first_sensor_value_col_index = 6
                            found_col_index = True
                            print("found header. parsing rows...")
                            continue

                        if not found_col_index:
                            try:
                                row_values = strip_list(row_values)
                            except AttributeError as e:
                                print_except(e)
                                date_col_index = 0
                                time_col_index = 1
                                control_station_col_index = 2
                                try:
                                    farm_id = int(row_values[control_station_col_index])
                                except ValueError as e:
                                    print(e)
                                except IndexError as e:
                                    print(e)
                                    continue
                                serial_number_col_index = 3
                                signal_strength_col_index = 4
                                battery_voltage_col_index = 5
                                first_sensor_value_col_index = 6
                                found_col_index = True
                                print("found header. parsing rows...")
                                continue
                                # exit()

                        # print(path)
                        # print(row_values)
                        # find index of each column
                        if not found_col_index:
                            try:
                                date_col_index = row_values.index("date")
                                time_col_index = row_values.index("time")
                                control_station_col_index = row_values.index(
                                    "controlstation"
                                )
                                serial_number_col_index = row_values.index(
                                    "tagserialnumber"
                                )
                                signal_strength_col_index = row_values.index(
                                    "signalstrength"
                                )
                                battery_voltage_col_index = row_values.index(
                                    "batteryvoltage"
                                )
                                first_sensor_value_col_index = row_values.index(
                                    "firstsensorvalue"
                                )
                                found_col_index = True

                                print("found header. parsing rows...")
                                continue
                            except ValueError as e:
                                if (
                                    "transpondernumber" in row_values
                                    and "activitylevel" in row_values
                                ):
                                    date_col_index = row_values.index("date")
                                    time_col_index = row_values.index("time")
                                    serial_number_col_index = row_values.index(
                                        "transpondernumber"
                                    )
                                    signal_strength_col_index = row_values.index(
                                        "signalstrength"
                                    )
                                    battery_voltage_col_index = row_values.index(
                                        "batterylife"
                                    )
                                    first_sensor_value_col_index = row_values.index(
                                        "activitylevel"
                                    )
                                    found_col_index = True

                                    print("found header. parsing rows...")
                                    continue
                                if (
                                    "transponderno" in row_values
                                    and "activitylevel" in row_values
                                ):
                                    date_col_index = row_values.index("date")
                                    time_col_index = row_values.index("time")
                                    serial_number_col_index = row_values.index(
                                        "transponderno"
                                    )
                                    first_sensor_value_col_index = row_values.index(
                                        "activitylevel"
                                    )
                                    if "signalstrength" in row_values:
                                        signal_strength_col_index = row_values.index(
                                            "signalstrength"
                                        )
                                    if "batterylife" in row_values:
                                        battery_voltage_col_index = row_values.index(
                                            "batterylife"
                                        )
                                    found_col_index = True

                                    print("found header. parsing rows...")
                                    continue
                                continue

                        # print("farm id is %d." % farm_id)
                        # print('progress %s/%d' % (curr_file, len(file_paths)))

                        try:
                            date_string = (
                                xl_date_to_date(row_values[date_col_index], book)
                                + " "
                                + convert_excel_time(row_values[time_col_index], book)
                            )
                            # print(date_string)
                        except TypeError as e:
                            print(e)
                            continue

                        epoch = int(
                            datetime.strptime(
                                date_string, "%d/%m/%Y %I:%M:%S %p"
                            ).timestamp()
                        )
                        control_station = (
                            farm_id  # int(row_values[control_station_col_index])
                        )
                        serial_number = int(row_values[serial_number_col_index])

                        try:
                            signal_strength = int(
                                str(row_values[signal_strength_col_index])
                                .replace("@", "")
                                .split(".")[0]
                            )
                        except (
                            IndexError,
                            NameError,
                            UnboundLocalError,
                            ValueError,
                        ) as e:
                            # print(e)
                            # print(path)
                            signal_strength = -1
                        try:
                            battery_voltage = int(
                                str(row_values[battery_voltage_col_index]).split(".")[
                                    0
                                ],
                                16,
                            )
                        except (
                            IndexError,
                            NameError,
                            UnboundLocalError,
                            ValueError,
                        ) as e:
                            # print(e)
                            # print(path)
                            battery_voltage = -1

                        try:
                            first_sensor_value = int(
                                row_values[first_sensor_value_col_index]
                            )
                        except ValueError as e:
                            print(e)
                            print(path)
                            continue

                        # print(curr_file, len(file_paths), farm_id, sheet.name, row_values, path,
                        #       first_sensor_value_col_index, battery_voltage_col_index, signal_strength_col_index,
                        #       serial_number_col_index, date_col_index, time_col_index)

                        record_log = "date_string=%s time=%s  row=%d epoch=%d control_station=%d serial_number=%d signal_strength=%d battery_voltage=%d first_sensor_value=%d" % (
                            date_string,
                            get_elapsed_time_string(start_time, time.time()),
                            valid_rows,
                            epoch,
                            control_station,
                            serial_number,
                            signal_strength,
                            battery_voltage,
                            first_sensor_value,
                        )
                        # print(record_log)

                        transponders[serial_number] = ""

                        df.append(
                            pd.DataFrame(
                                {
                                    "timestamp": epoch,
                                    "control_station": control_station,
                                    "serial_number": serial_number,
                                    "signal_strength": signal_strength,
                                    "battery_voltage": battery_voltage,
                                    "first_sensor_value": first_sensor_value,
                                },
                                index=[valid_rows],
                            )
                        )

                        valid_rows += 1
                    except ValueError as exception:
                        print_except(exception)
                        print(exception)
                        print(path)
                        if not None:
                            print(row_values)
                        log = "%d/%d--%s---%s---%s---%s" % (
                            curr_file,
                            len(file_paths),
                            get_elapsed_time_string(start_time, time.time()),
                            str(exception),
                            path,
                            record_log,
                        )
                        print(log)
                        log_file.write(log + "\n")

                print("transponders:", transponders.keys())
                if len(df) == 0:
                    continue

                store.append(
                    "/",
                    value=pd.concat(df),
                    format="t",
                    append=True,
                    data_columns=[
                        "timestamp",
                        "control_station",
                        "serial_number",
                        "signal_strength",
                        "battery_voltage",
                        "first_sensor_value",
                    ],
                )

                # del table_row
                # table_f.flush()
            del book
            del df
        except (ValueError, FileNotFoundError, xlrd.biffh.XLRDError) as e:
            print("error", e)
            print(path)
            continue
    store.close()


def main(
    raw_h5_filepath: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    db_name: str = "sb",
    create_db: bool = False,
    n_job: int = 2,
):
    """This script create a SQL database which will hold the activity data of the farm in multiple sampling resolution
    the resulted database is required for the webviewer\n
    Args:\n
        raw_h5_filepath: Raw data HDF5 file
        db_name: SQL database name
    """
    if create_db:
        create_and_connect_to_sql_db(str(db_name))
    # drop_all_tables(db_name)
    process_raw_h5files(str(raw_h5_filepath), njob=n_job)


if __name__ == "__main__":
    main(
        "E:/SouthAfrica/Tracking Data/raw_data.h5",
        "south_africa_activity",
        create_db=True
    )

    # main(
    #     "E:\SouthAfrica\Tracking Data\\Delmas\\raw_data_delmas_debug.h5",
    #     "south_africa",
    # )

    # main(
    #     "E:\SouthAfrica\Tracking Data\Cedara\\raw_data_cedara_debug.h5",
    #     "south_africa",
    # )

    # main(
    #     "F:\\Data2\\h5\\cedara.h5",
    #     "south_africa_cedara_test",
    # )

    # typer.run(main)

    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\Delmas", "raw_data_delmas.h5")
    # # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\Bothaville", "bothaville.h5")
    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\Cedara", "raw_data_cedara_debug.h5")
    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\Eenzaamheid", "raw_data_eenzaamheid_debug.h5")
    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\elandsberg", "raw_data_elandsberg_debug.h5")
    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\msinga", "raw_data_msinga_debug.h5")
    # generate_raw_files_from_xlsx("E:\SouthAfrica\Tracking Data\Cedara", "raw_data_cedara_debug.h5")
    # exit(-1)
