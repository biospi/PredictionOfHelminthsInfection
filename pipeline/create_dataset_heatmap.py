import argparse
import glob
import math
import sys
import matplotlib.patches as mpatches
import glob2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import scipy.stats
import seaborn as sns


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def parse_options(dataset_filepath):
    filename = dataset_filepath.split("/")[-1].replace(".csv", "")
    split = filename.split("_")
    farm_id = split[1] + "_" + split[2]
    sampling = split[5]
    threshi = int(split[7])
    threshz = int(split[9])
    day_before_famacha_test = int(split[4])
    return farm_id, sampling, threshi, threshz, day_before_famacha_test


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def resample(df, res="D"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(timestamp='first', date_str='first', first_sensor_value='sum'), skipna=True)
    return df_resampled


def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def process_activity_data(file, start_time, end_time, i, nfiles):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")

    entropy = scipy.stats.entropy(df_activity["first_sensor_value"].dropna())
    # entropy = entropy2(df_activity["first_sensor_value"].dropna().values)

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
    merge = activity.tolist() + [entropy, animal_id]
    return [animal_id, df_resampled, time, activity, entropy, merge]


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
    return data_frame


def create_annotation_matrix(df, time_axis, window_size):
    # annotation = np.random.rand(activity_list_matrix.shape[0], activity_list_matrix.shape[1])
    annotation = np.ones(df.iloc[:, :-3].shape) * -1
    missing_ids = []
    cpt = 0
    for i in range(annotation.shape[0]):
        a_id = df.iloc[i, :]['id']
        data_fam = np.array(df.iloc[i, :]['famacha'])
        if len(data_fam.shape) == 0:
            print("animal id %s in the activity data (csv files) do not exist in the famacha data." % a_id)
            annotation[i, :] = -2
            missing_ids.append(a_id)
            continue

        for f_data in data_fam:
            for j in range(time_axis.shape[0]):
                date_f = f_data[0].to_datetime64()
                target = f_data[1]
                days = int((date_f - time_axis[j]).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
                if days == 0:
                    annotation[i, j-window_size:j+1] = target
                    # df.iloc[i, j-window_size:j+1] = 10000000
                    cpt += 1
                    break

    return annotation, missing_ids


def add_famacha(row, fam_data):
    id = int(row["id"])
    if id in fam_data.keys():
        timestamps = fam_data[id]
        timestamps.sort(key=lambda x: x[0])
        row["famacha"] = timestamps
        row["id"] = str(row["id"]) + "  %06.2f" % row["entropy"]
    else:
        row["id"] = str(row["id"])[1:] + "*" + "  %06.2f" % row["entropy"]
    return row


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
    if len(files) == 0:
        raise IOError("missing activity files .csv! in %s" % args.activity_dir)
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
    #get FAMACHA data
    famacha_data = {}
    dataset = load_dataset(dataset_file)
    for i in range(dataset.shape[0]):
        row = dataset.iloc[i, :]
        date_of_famacha_test_str = row["dtf1"].strip("'")
        date_of_famacha_test = pd.to_datetime(date_of_famacha_test_str, format="%d/%m/%Y")
        label = row["label"]
        animal_id = row["serial"]
        if animal_id in famacha_data.keys():
            famacha_data[animal_id].append([date_of_famacha_test, label])
        else:
            famacha_data[animal_id] = [[date_of_famacha_test, label]]
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

    # activity_list = []
    time_axis = None
    # animal_ids = []
    entropy_list = []
    raw = []
    for res in results:
        item = res.get()
        # animal_ids.append(item[0])
        df_resampled = item[1]
        time_axis = item[2]
        # activity_list.append(item[3])
        entropy_list.append(item[4])
        raw.append(item[5])
    df_raw = pd.DataFrame(raw, dtype=object)
    header = [x for x in range(df_raw.shape[1])]
    header[-1] = "id"
    header[-2] = "entropy"
    df_raw.columns = header
    df_raw["famacha"] = np.nan
    df_raw = df_raw.apply(add_famacha, axis=1, args=(famacha_data,))

    enable_sort = True
    if enable_sort:
        df_raw = df_raw.sort_values(['entropy'], ascending=False, ignore_index=True)

    fig, ax = plt.subplots(figsize=(24.20, 10.80))
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    activity_list_matrix = df_raw.iloc[:, :-3].values

    time_axis_str = [pd.to_datetime(str(x)).strftime('%d/%m/%Y') for x in time_axis]
    ax.set_xticklabels(time_axis_str)

    n_x_ticks = ax.get_xticks().shape[0]
    labels_ = np.array(time_axis_str)[list(range(1, len(time_axis_str), int(len(time_axis_str) / n_x_ticks)))]
    labels_[0] = time_axis_str[0]
    labels_[-1] = time_axis_str[0]
    ax.set_xticklabels(labels_)

    annotation, missing_ids = create_annotation_matrix(df_raw, time_axis, day_before_famacha_test)

    im = ax.imshow(df_raw.iloc[:, :-3].values, cmap='gray', aspect='auto', interpolation="nearest", extent=[0, df_raw.iloc[:, :-3].values.shape[1], 0, df_raw.iloc[:, :-3].values.shape[0]])

    animal_ids_formatted_ent = df_raw["id"].values[::-1]
    ax.set_yticklabels(animal_ids_formatted_ent)
    ax.set_yticks(np.arange(len(animal_ids_formatted_ent)))

    for i in range(len(ax.get_yticklabels())):
        if "*" in str(animal_ids_formatted_ent[i]):
            ax.get_yticklabels()[i].set_color("tab:red")

    print("adding annotations...")
    for i in range(activity_list_matrix.shape[0]):
        for j in range(activity_list_matrix.shape[1]):
            an = int(annotation[i, j])
            if an < 0:
                continue
            color = "tab:blue"
            if an == 1:
                color = "tab:orange"
            #use ASCII 219 █ for text highlight instead of rectangle
            offset_y = 0.6
            offset_x = 0.9
            ax.text(j-offset_x, activity_list_matrix.shape[0] - i - offset_y, "█", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')

    param_str = "sampling=%s threshi=%d threshz=%d day_before_famacha_test=%d" % (sampling, threshi, threshz, day_before_famacha_test)
    ax.set_title("%s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d" % (farm_id, DATASET_INFO_STR, param_str, len(missing_ids), len(animal_ids_formatted_ent)))


    patch1 = mpatches.Patch(color='tab:blue', label=DATASET_INFO[7].replace("\n", ""))
    patch2 = mpatches.Patch(color='tab:orange', label=DATASET_INFO[8].replace("\n", ""))
    plt.legend(handles=[patch1, patch2], loc='lower left')

    ax.yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))

    fig.tight_layout()

    out_folder_path = args.output
    filename = "dataset_heatmap_%s_%s_%s.png" % (farm_id, param_str, enable_sort)
    create_rec_dir(out_folder_path)
    file_path = out_folder_path +"/"+ filename
    print(file_path)
    fig.savefig(file_path)
    fig.savefig(file_path.replace(".png", ".svg"))

    plt.show()

    print("done.")







