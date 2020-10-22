import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool


def parse_options(dataset_filepath):
    filename = dataset_filepath.split("/")[-1].replace(".csv", "")
    split = filename.split("_")
    farm_id = split[1] + "_" + split[2]
    sampling = split[5]
    threshi = int(split[7])
    threshz = int(split[9])
    day_before_famacha_test = split[4]
    return farm_id, sampling, threshi, threshz, day_before_famacha_test


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].split("_")[0])
    return animal_id


def resample(df, res="H"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(timestamp='first', date_str='first', first_sensor_value='sum'), skipna=True)
    return df_resampled


def process_activity_data(file, i, nfiles):
    print("processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")
    df_resampled = resample(df_activity)
    time = df_resampled.index.values
    activity = df_resampled.first_sensor_value.values
    return [animal_id, df_resampled, time, activity]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create heatmap of the heard with sample overlay.')
    parser.add_argument('output',
                        help='Directory path of heatmap output. (Directory will be created if does not exist)')
    parser.add_argument('activity_dir', help='Parent directory of the activity data.')
    parser.add_argument('dataset_file', help='Path of the datase file containing the samples.')
    parser.add_argument('--n_job', type=int, default=1, help='Number of thread to use.')
    args = parser.parse_args()

    print("Argument values:")
    print(args.output)
    print(args.activity_dir)
    print(args.dataset_file)
    print("njob=", args.n_job)

    farm_id, sampling, threshi, threshz, day_before_famacha_test = parse_options(args.dataset_file)
    print("farm_id=", farm_id)
    print("sampling=", sampling)
    print("threshi=", threshi)
    print("threshz=", threshz)
    print("day_before_famacha_test=", day_before_famacha_test)

    files = glob.glob(args.activity_dir+"/*.csv")
    files = [file.replace("\\", '/') for file in files]#prevent Unix issues

    pool = Pool(processes=args.n_job)
    results = []
    for i, file in enumerate(files):
        if 'median' in file or 'mean' in file:
            continue
        results.append(pool.apply_async(process_activity_data, (file, i, len(files))))
    pool.close()
    pool.join()
    pool.terminate()

    activity_list = []
    time_axis_list = []
    start_time = None
    end_time = None
    for res in results:
        animal_id = res.get()[0]
        df_resampled = res.get()[1]
        activity = res.get()[3]
        activity_list.append(activity)
        time = res.get()[2]
        time_axis_list.append(time)
        if start_time is None or start_time < time[0]:
            start_time = time[0]
        if end_time is None or end_time > time[0]:
            end_time = time[-1]
    


    print("done")







