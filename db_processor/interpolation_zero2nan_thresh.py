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
# from openpyxl import load_workbook
# from pycel import ExcelCompiler
# import cryptography #need to be imported or pip install cryptography
import pathlib
import sys
from multiprocessing import Pool
from sys import exit

import numpy as np
import pandas as pd

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
    mask["first_sensor_value"] = (df.groupby('new')['ones'].transform('count') <= thresh) | data[
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
    return df_1min


def zeros_to_nan(df_interpolated, zero_to_nan_threh):
    data = pd.DataFrame(df_interpolated["first_sensor_value"])
    bzidx = -1
    ezidx = -1
    difference = -1

    start_zero_count = False
    for i, value in enumerate(data["first_sensor_value"]):

        if start_zero_count:
            difference = i - bzidx
        if difference >= zero_to_nan_threh:
            ezidx = i
            data["first_sensor_value"][bzidx:ezidx] = np.nan
            #bzidx = i
            ezidx = -1
            start_zero_count = False
            difference = -1
        if value != 0 and start_zero_count == True:
            bzidx = -1
            ezidx = -1
            start_zero_count = False

        if value == 0 and start_zero_count == False:
            #if bzidx == -1:
            bzidx = i
            start_zero_count = True

    df_interpolated_zero2nan = df_interpolated
    df_interpolated_zero2nan["first_sensor_value"] = data
    return df_interpolated_zero2nan


def process_csv(output_directory, path, zero_to_nan_threh, interpolation_thesh, farm_id, animal_id):
    print("loading data...", path)
    df = pd.read_csv(path, sep=",")
    df_interpolated = thresholded_interpol(df.copy(), interpolation_thesh)
    data_zerofill_thresh = zeros_to_nan(df_interpolated, zero_to_nan_threh)
    export_rawdata_to_csv(output_directory, data_zerofill_thresh, farm_id, animal_id, interpolation_thesh, zero_to_nan_threh)


def export_rawdata_to_csv(output_directory, df, farm_id, animal_id, thresh_interpol, thresh_zero2nan):
    print("exporting data...")
    path = "%s/%s/interpolation_thesh_interpol_%d_zeros_%d/" % (output_directory, farm_id, thresh_interpol, thresh_zero2nan)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename_path = path + "%s_interpol_%d_zeros_%d.csv" % (animal_id, thresh_interpol, thresh_zero2nan)
    purge_file(filename_path)
    df.to_csv(filename_path, sep=',', index=False)
    print(filename_path)


if __name__ == '__main__':
    print("start...")
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # print(__location__)

    # csv_dir_path = "C:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\db_processor\\src\\csv_export\\backfill_1min\\delmas_70101200027\\*.csv"
    # zero_to_nan_threh = 60*4
    # interpolation_thesh = 60
    # n_process = 6
    print("arg: output_dir csv_dir_path zero_to_nan_threh interpolation_thesh n_process")
    if len(sys.argv) > 1:
        output_directory = sys.argv[1]
        csv_dir_path = sys.argv[2]
        zero_to_nan_threh = sys.argv[3]
        interpolation_thesh = sys.argv[4]
        n_process = int(sys.argv[5])
    else:
        exit(-1)

    files = glob.glob(csv_dir_path)
    print("found %d files." % len(files))

    MULTI_THREADING_ENABLED = (n_process > 1)

    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_process)
        for idx, csv_file in enumerate(files):
            farm_id = csv_file.split('\\')[-2]
            animal_id = csv_file.split('\\')[-1].replace('.csv', '')
            pool.apply_async(process_csv, (output_directory, csv_file, int(zero_to_nan_threh), int(interpolation_thesh), farm_id, animal_id,))
        pool.close()
        pool.join()
    else:
        for csv_file in files:
            farm_id = csv_file.split('\\')[-2]
            animal_id = csv_file.split('\\')[-1].replace('.csv','')
            process_csv(output_directory, csv_file, int(zero_to_nan_threh), int(interpolation_thesh), farm_id, animal_id)
