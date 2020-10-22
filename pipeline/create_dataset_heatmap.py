import argparse
import glob

import glob2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np


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


def resample(df, res="D"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(timestamp='first', date_str='first', first_sensor_value='sum'), skipna=False)
    return df_resampled


def process_activity_data(file, start_time, end_time, i, nfiles):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")

    #add herd start and end to create missing empty bins of full time range
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(start_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    df_activity = pd.concat([pd.DataFrame(data), df_activity], ignore_index=True)
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(end_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    df_activity = pd.concat([df_activity, pd.DataFrame(data)], ignore_index=True)

    df_resampled = resample(df_activity)
    time = df_resampled.index.values
    activity = df_resampled.first_sensor_value.values
    return [animal_id, df_resampled, time, activity]


def get_start_end_date(file, i, nfiles):
    print("get_start_end_date processing files %d/%d  ..." % (i, nfiles))
    df_activity = pd.read_csv(file, sep=",")
    df_activity.index = pd.to_datetime(df_activity.date_str)
    time = df_activity.index.values
    return [time[0], time[-1]]


def load_dataset(file):
    data_frame = pd.read_csv(file, sep=",")
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    hearder[-19] = "label"
    hearder[-18] = "elem_in_row"
    hearder[-17] = "date1"
    hearder[-16] = "date2"
    hearder[-15] = "serial"
    hearder[-14] = "famacha_score"
    hearder[-13] = "previous_famacha_score"
    hearder[-12] = "previous_famacha_score2"
    hearder[-11] = "previous_famacha_score3"
    hearder[-10] = "previous_famacha_score4"

    hearder[-9] = "dtf1"
    hearder[-8] = "dtf2"
    hearder[-7] = "dtf3"
    hearder[-6] = "dtf4"
    hearder[-5] = "dtf5"

    hearder[-4] = "nd1"
    hearder[-3] = "nd2"
    hearder[-2] = "nd3"
    hearder[-1] = "nd4"

    data_frame.columns = hearder

    dataset = data_frame.loc[data_frame['label'].isin(["True", "False"])].reset_index(drop=True)
    dataset = dataset.replace({"label": {'True': True, 'False': False}})
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create heatmap of the heard with sample overlay.')
    parser.add_argument('output',
                        help='Directory path of heatmap output. (Directory will be created if does not exist)')
    parser.add_argument('activity_dir', help='Parent directory of the activity data.')
    parser.add_argument('dataset_dir', help='Path of the directory containing dataset .csv and class info .tx.')
    parser.add_argument('--n_job', type=int, default=1, help='Number of thread to use.')
    args = parser.parse_args()

    print("Argument values:")
    print(args.output)
    print(args.activity_dir)
    print(args.dataset_dir)
    print("njob=", args.n_job)

    try:
        dataset_file = glob2.glob(args.dataset_dir+"/**/*.csv")[0].replace("\\", '/')
    except IndexError as e:
        print(e)
        print("could not find dataset (.csv) file in %s" % args.dataset_dir)
    info_files = glob2.glob(args.dataset_dir+"/**/*.txt")
    info_file = None
    for file in info_files:
        file = file.replace("\\", '/')
        if len(file.split('/')[-1]) > 50: #todo add filename lenght filter in config file or flag filename
            info_file = file
            print("found info file %s" % file)
            break
    if info_file is None:
        raise IOError("missing dataset info (.txt) file! in %s" % args.dataset_dir)
    with open(info_file) as f:
        DATASET_INFO = f.readlines()
        DATASET_INFO_STR = '|'.join(DATASET_INFO).replace("\n", "")

    farm_id, sampling, threshi, threshz, day_before_famacha_test = parse_options(dataset_file)
    print("farm_id=", farm_id)
    print("sampling=", sampling)
    print("threshi=", threshi)
    print("threshz=", threshz)
    print("day_before_famacha_test=", day_before_famacha_test)

    files = glob.glob(args.activity_dir+"/*.csv")
    files = [file.replace("\\", '/') for file in files]#prevent Unix issues

    #find start date and end date
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

    ######################################################
    pool = Pool(processes=args.n_job)
    results = []
    for i, file in enumerate(files):
        if 'median' in file or 'mean' in file:
            continue
        results.append(pool.apply_async(process_activity_data, (file, start_time, end_time, i, len(files))))
    pool.close()
    pool.join()
    pool.terminate()

    activity_list = []
    time_axis_list = []
    animal_ids = []
    for res in results:
        animal_ids.append(res.get()[0])
        df_resampled = res.get()[1]
        time_axis_list.append(res.get()[2])
        activity_list.append(res.get()[3])


    #get FAMACHA data
    do_ft = []
    target_list = []
    dataset = load_dataset(dataset_file)
    for i in range(dataset.shape[0]):
        row = dataset.iloc[i, :]
        date_of_famacha_test_str = row["dtf1"].strip("'")
        date_of_famacha_test = pd.to_datetime(date_of_famacha_test_str)
        label = row["label"]
        do_ft.append(date_of_famacha_test)
        target_list.append(label)

    activity_list_array = np.array(activity_list)
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    im = ax.imshow(activity_list_array)
    ax.set_xticks(np.arange(activity_list_array.shape[1]))
    ax.set_yticks(np.arange(len(animal_ids)))

    annotation = np.ones(activity_list_array.shape)
    print("adding annotations...")
    for i in range(len(animal_ids)):
        for j in range(activity_list_array.shape[1]):
            text = ax.text(j, i, str(annotation[i, j]),
                           ha="center", va="center", color="w")
            break
    ax.set_title("%s herd and dataset samples location\n%s" % (farm_id, DATASET_INFO_STR))
    fig.tight_layout()
    plt.show()

    print("done.")







