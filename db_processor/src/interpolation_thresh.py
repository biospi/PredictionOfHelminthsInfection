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

from classifier.src.herd_map import create_herd_map

sql_db = None
MAX_ACTIVITY_COUNT_BIO = 480
pd.set_option('display.max_columns', 48)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except (FileNotFoundError, PermissionError):
        print("file not found.")


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



def thresholded_interpol(df_1min, thresh):
    print("thresholded_interpol...", thresh)
    data = pd.DataFrame(df_1min["first_sensor_value"])
    mask = data.copy()
    df = pd.DataFrame(data["first_sensor_value"])
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask["first_sensor_value"] = (df.groupby('new')['ones'].transform('count') < thresh) | data[
        "first_sensor_value"].notnull()
    interpolated = data.interpolate().bfill()[mask]
    df_1min["first_sensor_value"] = interpolated
    histogram_array_nan_dur , histogram_array_no_activity_dur = [], []
    # clump = using_clump(data["first_sensor_value"].values)
    len_holes = [len(list(g)) for k, g in itertools.groupby(data["first_sensor_value"].values, lambda x: np.isnan(x)) if k]
    for nan_gap in len_holes:
        histogram_array_nan_dur.append(nan_gap)

    len_no_activity = [len(list(g)) for k, g in itertools.groupby(data["first_sensor_value"].values, lambda x: x == 0) if k]
    for zero_gap in len_no_activity:
        histogram_array_no_activity_dur.append(zero_gap)

    print("thresholded_interpol done.", thresh)

    # df_linterpolated = df_1min.copy()
    # df_linterpolated['signal_strength_2'] = df_linterpolated['signal_strength']
    # df_linterpolated['timestamp'] = df_linterpolated.index.values.astype(np.int64) // 10 ** 9
    # df_linterpolated['date_str'] = df_linterpolated.index.strftime('%Y-%m-%dT%H:%M')
    # df_linterpolated = df_linterpolated.assign(farm_id=df_linterpolated['farm_id'].max())
    #
    # df_linterpolated = df_linterpolated[['timestamp', 'date_str', 'serial_number', 'signal_strength', 'signal_strength_2', 'battery_voltage',
    #              'first_sensor_value']]
    #
    # df_linterpolated = df_linterpolated.assign(serial_number=df_linterpolated['serial_number'].max())  # fills in gap when agg fails because of empty sensor value
    #
    # df_linterpolated[['signal_strength', 'signal_strength_2', 'battery_voltage']] = df_linterpolated[['signal_strength', 'signal_strength_2', 'battery_voltage']].fillna(value=0)

    return df_1min


def process_csv(path, zero_to_nan_threh, interpolation_thesh, farm_id, animal_id):
    print("loading data...", path)
    df = pd.read_csv(path, sep=",")
    df_interpolated = thresholded_interpol(df.copy(), interpolation_thesh)
    export_rawdata_to_csv(df_interpolated, farm_id, animal_id, interpolation_thesh)



def export_rawdata_to_csv(df, farm_id, animal_id, thresh):
    print("exporting data...")
    path = "csv_export/interpolated_1min/%s/interpolation_thesh_%d/" % (farm_id, thresh)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename_path = path + "%s.csv" % animal_id
    purge_file(filename_path)
    df.to_csv(filename_path, sep=',', index=False)
    print(filename_path)


if __name__ == '__main__':
    print("start...")
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    print(__location__)
    csv_file = "C:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\db_processor\\src\\csv_export\\backfill_1min\\delmas_70101200027\\40101310026.csv"
    zero_to_nan_threh = 20
    interpolation_thesh = 3

    if len(sys.argv) > 1:
        print("arg: csv_file zero_to_nan_threh interpolation_thesh")
        csv_file = sys.argv[1]
        zero_to_nan_threh = sys.argv[2]
        interpolation_thesh = sys.argv[3]
    farm_id = csv_file.split('\\')[-2]
    animal_id = csv_file.split('\\')[-1].replace('.csv','')
    process_csv(csv_file, zero_to_nan_threh, interpolation_thesh, farm_id, animal_id)
