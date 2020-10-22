import argparse
import glob
import sys
import matplotlib.patches as mpatches
import glob2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os


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
    animal_id = int(file.split("/")[-1].split("_")[0])
    return animal_id


def resample(df, res="D"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(timestamp='first', date_str='first', first_sensor_value='sum'), skipna=True)
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


def create_annotation_matrix(activity_list_matrix, animal_ids, famacha_data, time_axis, window_size):
    # annotation = np.random.rand(activity_list_matrix.shape[0], activity_list_matrix.shape[1])
    annotation = np.ones(activity_list_matrix.shape) * -1
    missing_ids = []
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            a_id = animal_ids[i]
            try:
                for f_data in famacha_data[a_id]:
                    date_f = f_data[0].to_datetime64()
                    target = f_data[1]
                    days = int((date_f - time_axis[j]).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
                    if days == 0:
                        annotation[i, j-window_size:j+1] = target
            except KeyError as e:
                print("animal id %d in the activity data (csv files) do not exist in the famacha data." % a_id)
                annotation[i, :] = -2
                missing_ids.append(a_id)
                break
                # print(e)
    return annotation, missing_ids


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
    time_axis = None
    animal_ids = []
    for res in results:
        animal_ids.append(res.get()[0])
        df_resampled = res.get()[1]
        time_axis = res.get()[2]
        activity_list.append(res.get()[3])


    #get FAMACHA data
    famacha_data = {}
    dataset = load_dataset(dataset_file)
    for i in range(dataset.shape[0]):
        row = dataset.iloc[i, :]
        date_of_famacha_test_str = row["dtf1"].strip("'")
        date_of_famacha_test = pd.to_datetime(date_of_famacha_test_str)
        label = row["label"]
        animal_id = row["serial"]
        if animal_id in famacha_data.keys():
            famacha_data[animal_id].append([date_of_famacha_test, label])
        else:
            famacha_data[animal_id] = [[date_of_famacha_test, label]]

    activity_list_matrix = np.array(activity_list)
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    im = ax.imshow(activity_list_matrix, cmap='gray', aspect='auto', interpolation="nearest", extent=[0, activity_list_matrix.shape[1], 0, activity_list_matrix.shape[0]])

    time_axis_str = [pd.to_datetime(str(x)).strftime('%d/%m/%Y') for x in time_axis]
    ax.set_xticklabels(time_axis_str)

    n_x_ticks = ax.get_xticks().shape[0]
    labels_ = np.array(time_axis_str)[list(range(1, len(time_axis_str), int(len(time_axis_str) / n_x_ticks)))]
    labels_[0] = time_axis_str[0]
    labels_[-1] = time_axis_str[0]
    ax.set_xticklabels(labels_)

    annotation, missing_ids = create_annotation_matrix(activity_list_matrix.copy(), animal_ids, famacha_data, time_axis, day_before_famacha_test)

    animal_ids_formatted = ["%s*" % x if x in missing_ids else str(x) for x in animal_ids]

    ax.set_yticklabels(animal_ids_formatted)
    ax.set_yticks(np.arange(len(animal_ids_formatted)))

    for i in range(len(ax.get_yticklabels())):
        if "*" in str(animal_ids_formatted[i]):
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
            offset_y = 0.4
            offset_x = 0.9
            ax.text(j-offset_x, i+offset_y, "█", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')

    param_str = "sampling=%s threshi=%d threshz=%d day_before_famacha_test=%d" % (sampling, threshi, threshz, day_before_famacha_test)
    ax.set_title("%s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d" % (farm_id, DATASET_INFO_STR, param_str, len(missing_ids), len(animal_ids)))
    fig.tight_layout()

    patch1 = mpatches.Patch(color='tab:blue', label=DATASET_INFO[7].replace("\n", ""))
    patch2 = mpatches.Patch(color='tab:orange', label=DATASET_INFO[8].replace("\n", ""))
    plt.legend(handles=[patch1, patch2])

    ax.yaxis.set(ticks=np.arange(0.5, len(animal_ids)))

    out_folder_path = args.output
    filename = "dataset_heatmap_%s_%s.png" % (farm_id, param_str)
    create_rec_dir(out_folder_path)
    file_path = out_folder_path +"/"+ filename
    print(file_path)
    fig.savefig(file_path)
    fig.savefig(file_path.replace(".png", ".svg"))

    plt.show()

    print("done.")







