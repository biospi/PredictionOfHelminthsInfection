import argparse
from utils.Utils import create_rec_dir
from multiprocessing import Pool
import glob
import pandas as pd
import numpy as np


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def get_start_end_date(file, i, nfiles):
    print("get_start_end_date processing files %d/%d  ..." % (i, nfiles))
    df_activity = pd.read_csv(file, sep=",")
    df_activity.index = pd.to_datetime(df_activity.date_str)
    time = df_activity.index.values
    return [time[0], time[-1]]


def sum_(to_resample):
    # s = np.nan
    # if np.isnan(to_resample.values).any() != to_resample.size:
    #     s = np.sum(to_resample)
    # print(to_resample.shape)
    s = np.nan
    if to_resample.dropna().size > 0:
        s = np.sum(to_resample.dropna())
    return s


def resample(df, res="1T"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(timestamp="first", date_str="first", first_sensor_value=sum_,
                                             signal_strength='first', battery_voltage='first', xmin='first',
                                             xmax='first', ymin='first', ymax='first', zmin='first', zmax='first'))
    df_resampled["timestamp"] = df_resampled.index.values.astype(np.int64) // 10 ** 9
    df_resampled["date_str"] = df_resampled.index.strftime('%Y-%m-%dT%H:%M')
    return df_resampled


def process_activity_data(out_DIR, file, start_time, end_time, i, nfiles):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")

    #add herd start and end to create missing empty bins of full time range
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(start_time)).strftime('%Y-%m-%dT%H:%M'),
                    'first_sensor_value': np.nan, 'signal_strength': np.nan, 'battery_voltage': np.nan, 'xmin': np.nan,
                    'xmax': np.nan, 'ymin': np.nan, 'ymax': np.nan, 'zmin': np.nan, 'zmax': np.nan})
    df_activity = pd.concat([pd.DataFrame(data), df_activity], ignore_index=True)
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(end_time)).strftime('%Y-%m-%dT%H:%M'),
                    'first_sensor_value': np.nan, 'signal_strength': np.nan, 'battery_voltage': np.nan, 'xmin': np.nan,
                    'xmax': np.nan, 'ymin': np.nan, 'ymax': np.nan, 'zmin': np.nan, 'zmax': np.nan})
    df_activity = pd.concat([df_activity, pd.DataFrame(data)], ignore_index=True)
    df_resampled_activity = resample(df_activity)
    filename = file.split('/')[-1]
    out_filepath = out_DIR + "/" + filename
    df_resampled_activity.to_csv(out_filepath, sep=',', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='forces x axis on every animal trace.')
    parser.add_argument('output_dir',
                        help='output directory. (Directory will be created if does not exist)')
    parser.add_argument('activity_dir', help='Parent directory of the backfilled 1 min activity data.')
    parser.add_argument('--n_job', type=int, default=6, help='Number of thread to use.')
    args = parser.parse_args()

    print("Argument values:")
    print(args.output_dir)
    print(args.activity_dir)
    print("njob=", args.n_job)

    out_DIR = args.output_dir
    create_rec_dir(out_DIR)

    files = glob.glob(args.activity_dir+"/*.csv")
    if len(files) == 0:
        raise IOError("missing activity files .csv! in %s" % args.activity_dir)
    files = [file.replace("\\", '/') for file in files]#prevent Unix issues

    #find start date and end date##########################
    pool = Pool(processes=args.n_job)
    results_dates = []
    for i, file in enumerate(files):
        if 'median' in file or 'mean' in file:
            continue
        results_dates.append(pool.apply_async(get_start_end_date, (file, i, len(files))))
    pool.close()
    pool.join()
    pool.terminate()
    start_time = None
    end_time = None
    for res in results_dates:
        s = res.get()[0]
        e = res.get()[1]
        if start_time is None or s < start_time:
            start_time = s
        if end_time is None or e > end_time:
            end_time = e

    start_time = np.datetime64(str(start_time).split('T')[0])
    end_time = np.datetime64(str(end_time).split('T')[0])
    end_time = end_time + np.timedelta64(1, 'D')

    pool = Pool(processes=args.n_job)
    results = []
    for i, file in enumerate(files):
        results.append(pool.apply_async(process_activity_data, (out_DIR, file, start_time, end_time, i, len(files))))
    pool.close()
    pool.join()
    pool.terminate()