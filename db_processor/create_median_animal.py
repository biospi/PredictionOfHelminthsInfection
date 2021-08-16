import os.path
from sys import exit
import glob
import itertools
import os
import os.path
# import openpyxl
# import tables
# from cassandra.cluster import Cluster
# from ipython_genutils.py3compat import xrange
# from tables import *
import os.path
import sys
from sys import exit

import numpy as np
import pandas as pd

# from openpyxl import load_workbook
# from pycel import ExcelCompiler
# import cryptography #need to be imported or pip install cryptography


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
    #print("thresholded_interpol...", thresh)
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


def export_rawdata_to_csv(df, dir_path, thresh_i, thresh_zero2nan):
    print("exporting data...")
    filename_path = dir_path + "median_thesh_interpol_%d_zeros_%d.csv" % (thresh_i, thresh_zero2nan)
    purge_file(filename_path)
    df.to_csv(filename_path, sep=',', index=False)
    print(filename_path)


if __name__ == '__main__':
    print("start...")
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # print(__location__)
    print("arg: csv_dir")
    if len(sys.argv) > 1:
        csv_dir = sys.argv[1]
    else:
        exit(-1)

    df_raw = pd.DataFrame()

    files = glob.glob(csv_dir)
    if len(files) == 0:
        print("no files in %s" % csv_dir)
        exit(-1)

    thresh_i = None
    thresh_zero2nan = None
    for idx, file in enumerate(files):
        print(file)
        if 'median' in file.split('\\')[-1]:
            continue
        farm_id = file.split('\\')[-2]
        animal_id = file.split('\\')[-1].replace('.csv', '')
        thresh_i = int(animal_id.split('_')[2])
        thresh_zero2nan = int(animal_id.split('_')[4])
        df = pd.read_csv(file, sep=",")
        df_raw[str(idx)] = df['first_sensor_value']

    compute_median = pd.DataFrame(df_raw.median(axis=1, skipna=True))
    compute_median["timestamp"] = df["timestamp"]
    compute_median["date_str"] = df["date_str"]
    compute_median = compute_median.rename(columns={0: "first_sensor_value"})
    compute_median = compute_median[["timestamp", "date_str", "first_sensor_value"]]

    # for debugging
    # for i, col in enumerate(df_raw.columns):
    #     df_raw[col][200000: 200300].plot(alpha=0.5)
    # compute_median["first_sensor_value"][200000: 200300].plot(color="black")
    # plt.show()

    export_rawdata_to_csv(compute_median, csv_dir[:-5], thresh_i, thresh_zero2nan)
