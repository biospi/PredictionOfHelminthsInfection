import os.path
import shutil
from sys import exit
import glob
import itertools
import os
import os.path
import os.path
import pathlib
import sys
from multiprocessing import Pool
from sys import exit

import numpy as np
import pandas as pd


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except:
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


def inputation_interpol(df_1min, thresh, imputation_type=0):
    #print("thresholded_interpol...", thresh)
    data = pd.DataFrame(df_1min["first_sensor_value"])
    mask = data.copy()
    df = pd.DataFrame(data["first_sensor_value"])
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask["first_sensor_value"] = (df.groupby('new')['ones'].transform('count') <= thresh) | data[
        "first_sensor_value"].notnull()
    if imputation_type == 1:
        padded = data.fillna(-1).bfill()[mask]
        df_1min["first_sensor_value"] = padded
    if imputation_type == 0:
        interpolated = data.interpolate().bfill()[mask]
        df_1min["first_sensor_value"] = interpolated

    # len_holes = [len(list(g)) for k, g in itertools.groupby(data["first_sensor_value"].values, lambda x: np.isnan(x)) if k]
    # for nan_gap in len_holes:
    #     histogram_array_nan_dur.append(nan_gap)
    #
    # len_no_activity = [len(list(g)) for k, g in itertools.groupby(data["first_sensor_value"].values, lambda x: x == 0) if k]
    # for zero_gap in len_no_activity:
    #     histogram_array_no_activity_dur.append(zero_gap)

    print("thresholded_interpol done.", thresh)
    return df_1min


def zeros_to_nan(df_imputated, zero_to_nan_threh):
    data = pd.DataFrame(df_imputated["first_sensor_value"])
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
            # bzidx = i
            ezidx = -1
            start_zero_count = False
            difference = -1
        if value != 0 and start_zero_count == True:
            bzidx = -1
            ezidx = -1
            start_zero_count = False

        if value == 0 and start_zero_count == False:
            # if bzidx == -1:
            bzidx = i
            start_zero_count = True

    df_imputated_zero2nan = df_imputated
    df_imputated_zero2nan["first_sensor_value"] = data
    return df_imputated_zero2nan


def process_csv(output_directory, path, zero_to_nan_threh, interpolation_thesh, farm_id, animal_id,
                imputation_type):
    print("loading data...", path)
    df = pd.read_csv(path, sep=",")

    # if imputation_type == 0:
    #     df_imputated = inputation_interpol(df.copy(), interpolation_thesh, imputation_type=0)
    #     data_zerofill_thresh = zeros_to_nan(df_imputated, zero_to_nan_threh)
    #
    # if imputation_type == 1:
    #     df = df.fillna(0)
    #     data_zerofill_thresh = zeros_to_nan(df, zero_to_nan_threh)

    # path = export_rawdata_to_csv(output_directory, data_zerofill_thresh, farm_id, animal_id, interpolation_thesh,
    #                       zero_to_nan_threh, imputation_type)
    return path


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def export_rawdata_to_csv(output_directory, df, farm_id, animal_id, thresh_interpol, thresh_zero2nan,
                          imputation_type):
    print("exporting data...")
    print("output_directory=", output_directory)
    # try:
    #     pathlib.Path(output_directory).mkdir(parents=True)
    # except:
    #     pass
    print("farm_id=", farm_id)
    path = "%s/%s_interpol_%d_zeros_%d_imputation_%d/" % (
        output_directory, farm_id, thresh_interpol, thresh_zero2nan, imputation_type)
    # # print("path=",path)
    # # try:
    # #     pathlib.Path(path).mkdir(parents=True)
    # # except:
    # #     pass
    # path = path + "interpol_%d_zeros_%d_%s/" %(thresh_interpol, thresh_zero2nan, farm_id)
    # # print("path2=", path)
    # # try:
    # #     pathlib.Path(path).mkdir()
    # #     print("created path2")
    # # except Exception as e:
    # #     print(e)
    create_rec_dir(path)
    filename_path = path + "%s_interpol_%d_zeros_%d_farmid_%s_imputation_%d.csv" % (
        animal_id, thresh_interpol, thresh_zero2nan, farm_id, imputation_type)
    # print("purge_file")
    # purge_file(filename_path)
    print("create_file=", filename_path)
    df.to_csv(filename_path, sep=',', index=False)
    print("final file=", filename_path)
    return path


def parse_filename(file):
    farm_id = file.split('/')[-2]
    animal_id = file.split('/')[-1].replace('.csv', '')
    return farm_id, animal_id


if __name__ == '__main__':
    print("arg: output_dir csv_dir_path interpolation_thesh zero_to_nan_threh imputation_type n_process")
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        csv_dir_path = sys.argv[2]
        interpolation_thesh = sys.argv[3]
        zero_to_nan_threh = sys.argv[4]
        imputation_type = int(sys.argv[5])
        n_process = int(sys.argv[6])
    else:
        exit(-1)

    print("imputation_type=", imputation_type)
    if imputation_type not in [0, 1]:
        raise ValueError("imputatation type must be 0(linear interpolation) or 1(-1 padding)!")

    files = glob.glob(csv_dir_path + "/*.csv")
    print("found %d files." % len(files))

    MULTI_THREADING_ENABLED = (n_process > 1)
    print("MULTI_THREADING_ENABLED=", MULTI_THREADING_ENABLED)

    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_process)
        for idx, csv_file in enumerate(files):
            csv_file = csv_file.replace("\\", "/")
            farm_id, animal_id = parse_filename(csv_file)
            result = pool.apply_async(process_csv, (
                output_dir, csv_file, int(zero_to_nan_threh), int(interpolation_thesh), farm_id, animal_id,
                imputation_type,))
        pool.close()
        pool.join()
        pool.terminate()
        out_dir = result.get()
    else:
        for csv_file in files:
            csv_file = csv_file.replace("\\", "/")
            farm_id, animal_id = parse_filename(csv_file)
            out_dir = process_csv(output_dir, csv_file, int(zero_to_nan_threh), int(interpolation_thesh), farm_id, animal_id,
                        imputation_type)


    df_raw = pd.DataFrame()
    print("csv_dir=", out_dir)
    files = glob.glob(out_dir + "/*.csv")
    if len(files) == 0:
        print("no files in %s" % out_dir)
        exit(-1)

    thresh_i = None
    thresh_zero2nan = None
    for idx, file in enumerate(files):
        file = file.replace("\\", '/')
        print("file=", file)
        if 'median' in file.split('/')[-1] or 'mean' in file.split('/')[-1]:
            continue
        split = file.split('/')[-1].replace('.csv', '').split("_")
        animal_id = int(split[0])
        farm_id = split[6] + "_" + split[7]
        thresh_i = int(split[2])
        thresh_zero2nan = int(split[4])
        df = pd.read_csv(file, sep=",")
        df_raw[str(idx)] = df['first_sensor_value']

    compute_mean = pd.DataFrame(df_raw.mean(axis=1, skipna=True))
    compute_mean["timestamp"] = df["timestamp"]
    compute_mean["date_str"] = df["date_str"]
    compute_mean = compute_mean.rename(columns={0: "first_sensor_value"})
    compute_mean = compute_mean[["timestamp", "date_str", "first_sensor_value"]]

    compute_median = pd.DataFrame(df_raw.median(axis=1, skipna=True))
    compute_median["timestamp"] = df["timestamp"]
    compute_median["date_str"] = df["date_str"]
    compute_median = compute_median.rename(columns={0: "first_sensor_value"})
    compute_median = compute_median[["timestamp", "date_str", "first_sensor_value"]]

    print("exporting data...")
    print("dir_path=", out_dir)
    filename_path = out_dir + "/mean_interpol_%d_zeros_%d_farmid_%s_imputation_%d.csv" % (thresh_i, thresh_zero2nan,
                                                                                          farm_id, imputation_type)
    purge_file(filename_path)
    compute_mean.to_csv(filename_path, sep=',', index=False)
    print("output file=", filename_path)

    print("exporting data...")
    print("dir_path=", out_dir)
    filename_path = out_dir + "/median_interpol_%d_zeros_%d_farmid_%s_imputation_%d.csv" % (thresh_i, thresh_zero2nan,
                                                                                            farm_id, imputation_type)
    purge_file(filename_path)
    compute_median.to_csv(filename_path, sep=',', index=False)
    print("output file=", filename_path)