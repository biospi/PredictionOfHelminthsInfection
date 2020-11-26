import os
import os.path
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from sys import exit

import dateutil.relativedelta
import numpy as np
import pandas as pd
import tables

MAX_ACTIVITY_COUNT_BIO = 480


def get_elapsed_time_string(time_initial, time_next):
    dt1 = datetime.fromtimestamp(time_initial)
    dt2 = datetime.fromtimestamp(time_next)
    rd = dateutil.relativedelta.relativedelta(dt2, dt1)
    return '%02d:%02d:%02d:%02d' % (rd.days, rd.hours, rd.minutes, rd.seconds)


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


def process_raw_h5files(path, output_dir, n_job):
    print(path)
    h5_raw = tables.open_file(path, "r")
    data = h5_raw.root.table
    list_raw = []
    print("loading data...")
    cpt = 0
    cpt_tot = 0
    for idx, x in enumerate(data):  # need idx for data iteration?
        cpt_tot += 1
        farm_id = x['control_station']

        # if str(x['serial_number'])[-3:] != '125':
        #     continue
        #
        # d = datetime.fromtimestamp(x['timestamp']).strftime("%Y-%m-%dT%H:%M:%S")
        # print(d)
        # if "2015-02-15T18:05" in d:
        #     print("debug")

        if x['first_sensor_value'] > MAX_ACTIVITY_COUNT_BIO or x['first_sensor_value'] < 0:
            print("out of bound value!!!", x['first_sensor_value'])
            cpt += 1
            continue
        value = (x['timestamp'], farm_id, x['serial_number'], x['signal_strength'], x['battery_voltage'],
                 x['first_sensor_value'], datetime.fromtimestamp(x['timestamp']).strftime("%Y-%m-%dT%H:%M:%S"),
                 datetime.strptime(datetime.fromtimestamp(x['timestamp']).strftime("%Y-%m-%dT%H:%M:%S"),
                                   '%Y-%m-%dT%H:%M:%S'), x['xmin'], x['xmax'], x['ymin'], x['ymax'], x['zmin'], x['zmax'])
        list_raw.append(value)
        # if idx > 100000:  # todo remove
        #     break
    # group records by farm id/control_station
    print("FOUND %d OUT OF BOUND VALUES / %d" % (cpt, cpt_tot))
    groups = defaultdict(list)
    for i, obj in enumerate(list_raw):
        groups[obj[1]].append(obj)
    animal_list_grouped_by_farmid = list(groups.values())

    for group in animal_list_grouped_by_farmid:
        farm_id = str(group[0][1])
        process_raw_file(farm_id, group, output_dir, n_job)


def create_dataframe(list_data):
    df = pd.DataFrame(list_data)
    df.columns = ['timestamp', 'farm_id', 'serial_number', 'signal_strength', 'battery_voltage', 'first_sensor_value',
                  'date', 'date_str', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
    df = df.drop_duplicates(subset=['timestamp', 'first_sensor_value'])
    df = df.sort_values(by='date')
    df.rename({'date': 'index'}, axis=1, inplace=True)
    return df


def isNaN(num):
    return num != num


def resample_1min(df_raw):
    start = time.time()
    print("resample_1min...")
    df_raw = df_raw.set_index('index')
    df_raw.index.name = None
    df_raw.index = pd.to_datetime(df_raw.index)
    start_date = min(df_raw.index.array).replace(second=0)
    end_date = max(df_raw.index.array).replace(second=0)
    date_range = np.array(pd.date_range(start_date, end_date, freq='1T', normalize=False))
    df_1min = pd.DataFrame(date_range)
    df_1min.index = date_range
    df_raw.index.name = 'index'
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
        if isNaN(df_1min.at[df_1min.index[bin], 'timestamp']):
            df_1min.at[df_1min.index[bin], 'timestamp'] = row_raw.timestamp
            df_1min.at[df_1min.index[bin], 'farm_id'] = row_raw.farm_id
            df_1min.at[df_1min.index[bin], 'serial_number'] = row_raw.serial_number
            df_1min.at[df_1min.index[bin], 'signal_strength'] = row_raw.signal_strength
            df_1min.at[df_1min.index[bin], 'battery_voltage'] = row_raw.battery_voltage
            df_1min.at[df_1min.index[bin], 'first_sensor_value'] = row_raw.first_sensor_value
            df_1min.at[df_1min.index[bin], 'date_str'] = row_raw.date_str
            df_1min.at[df_1min.index[bin], 'xmin'] = row_raw.xmin
            df_1min.at[df_1min.index[bin], 'xmax'] = row_raw.xmax
            df_1min.at[df_1min.index[bin], 'ymin'] = row_raw.ymin
            df_1min.at[df_1min.index[bin], 'ymax'] = row_raw.ymax
            df_1min.at[df_1min.index[bin], 'zmin'] = row_raw.zmin
            df_1min.at[df_1min.index[bin], 'zmax'] = row_raw.zmax

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
                if isNaN(df_1min.at[df_1min.index[bin], 'timestamp']):
                    df_1min.at[df_1min.index[bin], 'timestamp'] = row_raw.timestamp
                    df_1min.at[df_1min.index[bin], 'farm_id'] = row_raw.farm_id
                    df_1min.at[df_1min.index[bin], 'serial_number'] = row_raw.serial_number
                    df_1min.at[df_1min.index[bin], 'signal_strength'] = row_raw.signal_strength
                    df_1min.at[df_1min.index[bin], 'battery_voltage'] = row_raw.battery_voltage
                    df_1min.at[df_1min.index[bin], 'first_sensor_value'] = row_raw.first_sensor_value
                    df_1min.at[df_1min.index[bin], 'date_str'] = row_raw.date_str
                    df_1min.at[df_1min.index[bin], 'xmin'] = row_raw.xmin
                    df_1min.at[df_1min.index[bin], 'xmax'] = row_raw.xmax
                    df_1min.at[df_1min.index[bin], 'ymin'] = row_raw.ymin
                    df_1min.at[df_1min.index[bin], 'ymax'] = row_raw.ymax
                    df_1min.at[df_1min.index[bin], 'zmin'] = row_raw.zmin
                    df_1min.at[df_1min.index[bin], 'zmax'] = row_raw.zmax
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


def build_data_from_raw(farm_id, animal_records_df, output_dir, has_xyz=True):
    df_raw_cleaned = resample_1min(animal_records_df)
    animal_id = int(df_raw_cleaned['serial_number'][df_raw_cleaned['serial_number'].notnull()].values[0])
    if has_xyz:
        df_raw_cleaned_sub = df_raw_cleaned[['timestamp', 'date_str', 'first_sensor_value', 'signal_strength', 'battery_voltage', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']]
    else:
        df_raw_cleaned_sub = df_raw_cleaned[['timestamp', 'date_str', 'first_sensor_value', 'signal_strength', 'battery_voltage']]

    df_raw_cleaned_sub_fill = df_raw_cleaned_sub.copy()
    df_raw_cleaned_sub_fill['timestamp'] = df_raw_cleaned_sub_fill.index.values.astype(np.int64) // 10 ** 9
    df_raw_cleaned_sub_fill['date_str'] = df_raw_cleaned_sub_fill.index.strftime('%Y-%m-%dT%H:%M')

    export_rawdata_to_csv(df_raw_cleaned_sub_fill, farm_id, animal_id, output_dir)
    return df_raw_cleaned_sub_fill


def process(farm_id, animal_records, output_dir):
    animal_records_df = create_dataframe(animal_records)
    build_data_from_raw(farm_id, animal_records_df, output_dir)


def export_data_to_csv(data, farm_id):
    print("exporting data...")
    filename = "%s_1min.csv" % (farm_id)
    df = pd.DataFrame(data)
    df.columns = ['epoch', 'datetime', 'animal_id', 's1', 's2', 'battery', 'activity']
    df_sub = df[['epoch', 'datetime', 'animal_id', 'activity']]
    df_sub.to_csv(filename, sep=',', index=False)
    print("filename=", filename)


def export_rawdata_to_csv(df, farm_id, animal_id, output_dir):
    print("exporting data...")
    path = "%s/%s/" % (output_dir, farm_id)
    print("out_dir=", path)
    if not os.path.exists(path):
        print("mkdir", path)
        os.makedirs(path)

    filename_path = path + "/%s.csv" % animal_id
    df.to_csv(filename_path, sep=',', index=False)
    print("export=", filename_path)


def process_raw_file(farm_id, data, output_dir, n_job):
    start_time = time.time()
    farm_id = format_farm_id(farm_id)
    print("process data for farm %s." % farm_id)

    # group records by animal id/serial number
    groups = defaultdict(list)
    for obj in data:
        groups[obj[2]].append(obj)
    animal_list_grouped_by_serialn = list(groups.values())

    MULTI_THREADING_ENABLED = (n_job > 0)
    MIN_RECORD_NUMBER = 1
    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_job)
        for idx, animal_records in enumerate(animal_list_grouped_by_serialn):
            if len(animal_records) <= MIN_RECORD_NUMBER: #if animal have less than 24 hours of data dismiss
                continue
            print("animal_records=", len(animal_records))
            pool.apply_async(process, (farm_id, animal_records, output_dir,))
        pool.close()
        pool.join()
    else:
        for idx, animal_records in enumerate(animal_list_grouped_by_serialn):
            print("progress=%d/%d." % (idx, len(animal_list_grouped_by_serialn)))
            if len(animal_records) <= MIN_RECORD_NUMBER:
                continue
            print("animal_records=", len(animal_records))
            process(farm_id, animal_records, output_dir)

    print(get_elapsed_time_string(start_time, time.time()))
    print("finished processing raw file.")


if __name__ == '__main__':
    print("arg: output_dir raw_h5_filepath n_job")
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        raw_h5_filepath = sys.argv[2]
        n_job = int(sys.argv[3])
    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("raw_h5_filepath=", raw_h5_filepath)
    print("n_job=", n_job)

    process_raw_h5files(raw_h5_filepath, output_dir, n_job)
