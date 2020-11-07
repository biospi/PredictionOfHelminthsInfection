import argparse
import glob
import json
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
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates


def breaklineinsert_(str):
    midPoint = len(str)//2
    return str[:midPoint] + '\n' + str[midPoint:]


def breaklineinsert(str):
    midPoint = len(str)//2
    return breaklineinsert_(str[:midPoint]) + '\n' + breaklineinsert_(str[midPoint:])


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
    day_before_famacha_test = int(split[4])
    return farm_id, sampling, day_before_famacha_test


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def entropy_(to_resample):
    e = np.nan
    if to_resample.dropna().size != 0:
        #input_array = to_resample.dropna()
        input_array = to_resample.fillna(1)
        e = scipy.stats.entropy(input_array)
        # print(to_resample.size)
    return e


def sum_(to_resample):
    # s = np.nan
    # if np.isnan(to_resample.values).any() != to_resample.size:
    #     s = np.sum(to_resample)
    # print(to_resample.shape)
    s = np.nan
    if to_resample.dropna().size > 0:
        s = np.sum(to_resample)
    return s


def median_(to_resample):
    m = np.nan
    if to_resample.dropna().size > 0:
        to_resample = to_resample.fillna(-1)
        m = np.median(to_resample)
        if m == -1:
            m = np.nan
    return m


def resample(df, res="1D"):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(first_sensor_value=sum_))
    df_resampled_entropy = df.resample(res).agg(dict(first_sensor_value=entropy_))
    df_resampled_median = df.resample(res).agg(dict(first_sensor_value=median_))
    return df_resampled, df_resampled_entropy, df_resampled_median


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
    if np.isnan(entropy):
        entropy = 0
    # entropy = entropy2(df_activity["first_sensor_value"].dropna().values)

    #add herd start and end to create missing empty bins of full time range
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(start_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    df_activity = pd.concat([pd.DataFrame(data), df_activity], ignore_index=True)
    data = []
    data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(end_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    df_activity = pd.concat([df_activity, pd.DataFrame(data)], ignore_index=True)

    df_resampled_activity, df_resampled_entropy, df_resampled_median = resample(df_activity)
    time = df_resampled_activity.index.values
    activity = df_resampled_activity.first_sensor_value.values
    activity_e = df_resampled_entropy.first_sensor_value.values
    activity_m = df_resampled_median.first_sensor_value.values

    merge_a = activity.tolist() + [entropy, animal_id]
    merge_e = activity_e.tolist() + [entropy, animal_id]
    merge_m = activity_m.tolist() + [entropy, animal_id]
    return [animal_id, df_resampled_activity, df_resampled_entropy, time, activity, entropy, merge_a, merge_e, merge_m]


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
    # annotation = np.ones(df.iloc[:, :-4].shape) * -1
    annotation = np.empty(df.iloc[:, :-4].shape, dtype="<U10")
    missing_ids = []
    cpt = 0
    for i in range(annotation.shape[0]):
        a_id = df.iloc[i, :]['id']
        data_fam = np.array(df.iloc[i, :]['famacha'])
        if len(data_fam.shape) == 0:
            print("animal id %s in the activity data (csv files) do not exist in the famacha data." % a_id)
            annotation[i, :] = "n"
            missing_ids.append(a_id)
            continue

        for f_data in data_fam:
            for j in range(time_axis.shape[0]):
                date_f = f_data[0].to_datetime64()
                target = f_data[1]
                days = int((date_f - time_axis[j]).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
                if days == 0:
                    annotation[i, j-window_size:j] = target
                    # df.iloc[i, j-window_size:j+1] = 10000000
                    cpt += 1
                    break

    return annotation, missing_ids


def add_famacha_format_id(row, fam_data):
    id = int(row["id"])
    if id in fam_data.keys():
        timestamps = fam_data[id]
        timestamps.sort(key=lambda x: x[0])
        row["famacha"] = timestamps
        row["id"] = str(row["id"]) + "  %06.2f" % row["entropy"]
    else:
        row["id"] = str(row["id"])[1:] + "*" + "  %06.2f" % row["entropy"]
    return row


def export_tranponder_traces(row, out_dir, farm_id, time_axis, i_c, i_t):
    print("export_tranponder_traces %d/%d..." % (i_c, i_t))
    export_dir = out_dir + "/transponder_export"
    create_rec_dir(export_dir)
    plt.clf()
    activity = row[:-4].values
    entropy = row["entropy"]
    id = row["id"].split(" ")[0]
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.bar(time_axis, activity)
    xlabel = "Time (days)"
    ylabel = "Activity count"
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title("%s activity output of transponder %s entropy of entire trace=%.4f" % (farm_id, id, entropy))
    #plt.show()
    filename = "%s_%.4f_%s" % (farm_id, entropy, id)
    filename = filename.replace(".", "_").replace("*", "") + ".png"
    filepath = "%s/%s" % (export_dir, filename)
    print('saving fig...')
    fig.savefig(filepath)
    print("saved!")
    fig.clear()
    plt.close(fig)


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
        dataset_file_path = glob2.glob(args.dataset_dir + "/**/*.csv")[0].replace("\\", '/')
    except IndexError as e:
        print(e)
        print("could not find dataset (.csv) file in %s" % args.dataset_dir)
    info_files = glob2.glob(args.dataset_dir+"/**/*.json")
    info_file = None
    for file in info_files:
        file = file.replace("\\", '/')
        info_file = file
        print("found info file %s" % file)
        break
    if info_file is None:
        raise IOError("missing dataset info (.json) file! in %s" % args.dataset_dir)
    DATASET_INFO = json.load(open(info_file))

    farm_id, sampling, day_before_famacha_test = parse_options(dataset_file_path)

    print("exporting individual traces...")
    out_DIR = args.output + "/" + farm_id
    create_rec_dir(out_DIR)

    print("farm_id=", farm_id)
    print("sampling=", sampling)
    print("day_before_famacha_test=", day_before_famacha_test)

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
    ######################################################
    #get FAMACHA data
    famacha_data = {}
    dataset = load_dataset(dataset_file_path)
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
    raw_e = []
    raw_m = []
    for res in results:
        item = res.get()
        # animal_ids.append(item[0])
        df_resampled_a = item[1]
        df_resampled_e = item[2]
        time_axis = item[3]
        # activity_list.append(item[3])
        entropy_list.append(item[5])
        raw.append(item[6])
        raw_e.append(item[7])
        raw_m.append(item[8])
    df_raw = pd.DataFrame(raw, dtype=object)
    df_raw_e = pd.DataFrame(raw_e, dtype=object)
    df_raw_m = pd.DataFrame(raw_m, dtype=object)

    header = [x for x in range(df_raw.shape[1])]
    header[-1] = "id"
    header[-2] = "entropy"

    df_raw.columns = header
    df_raw["famacha"] = np.nan
    df_raw = df_raw.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    df_raw["possible"] = ['*' in x for x in df_raw["id"].values]
    df_raw = df_raw.sort_values(['possible', 'entropy'], ascending=[True, False], ignore_index=True).groupby('possible').head(df_raw.shape[0])
    # df_raw = df_raw.sort_values(['entropy'], ascending=False, ignore_index=True)


    df_raw_e.columns = header
    df_raw_e["famacha"] = np.nan
    df_raw_e = df_raw_e.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    df_raw_e["possible"] = ['*' in x for x in df_raw_e["id"].values]
    df_raw_e = df_raw_e.sort_values(['possible', 'entropy'], ascending=[True, False], ignore_index=True).groupby('possible').head(df_raw.shape[0])
    # df_raw_e = df_raw_e.sort_values(['entropy'], ascending=False, ignore_index=True)

    df_raw_m.columns = header
    df_raw_m["famacha"] = np.nan
    df_raw_m = df_raw_m.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    df_raw_m["possible"] = ['*' in x for x in df_raw_m["id"].values]
    df_raw_m = df_raw_m.sort_values(['possible', 'entropy'], ascending=[True, False], ignore_index=True).groupby('possible').head(df_raw.shape[0])

    df_raw = df_raw[df_raw["possible"] == False]
    df_raw_e = df_raw_e[df_raw_e["possible"] == False]
    df_raw_m = df_raw_m[df_raw_m["possible"] == False]
    #########################################################################

    # pool = Pool(processes=args.n_job)
    # for index, row in df_raw.iterrows():
    #     pool.apply_async(export_tranponder_traces, (row, out_DIR, farm_id, time_axis, index, df_raw.shape[0], ))
    # pool.close()
    # pool.join()
    # pool.terminate()
    # for index, row in df_raw.iterrows():
    #     export_tranponder_traces(row, out_DIR, farm_id, time_axis, index, df_raw.shape[0])
    ##########################################################################

    n = 3
    h = (df_raw.shape[0] * 20 * n) / 100
    w = 36.20 * 1
    fig, axs = plt.subplots(n, figsize=(w, h))
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()
    axs[2].yaxis.set_label_position("right")

    # axs[0].set_xticks(time_axis)
    # axs[1].set_xticks(time_axis)
    # axs[2].set_xticks(time_axis)
    # time_axis_str = [pd.to_datetime(str(x)).strftime('%d/%b/%Y') for x in time_axis]
    #
    # axs[0].set_xticklabels(time_axis_str)
    # axs[1].set_xticklabels(time_axis_str)
    # axs[2].set_xticklabels(time_axis_str)

    # n_x_ticks = axs[0].get_xticks().shape[0]
    # labels_ = np.array(time_axis_str)[list(range(1, len(time_axis_str), int(len(time_axis_str) / n_x_ticks)))]
    # # axs[0].tick_params(axis='x', rotation=90)
    # # axs[1].tick_params(axis='x', rotation=90)
    # # axs[2].tick_params(axis='x', rotation=90)
    # # labels_[0] = time_axis_str[0]
    # # labels_[-1] = time_axis_str[0]
    # axs[0].set_xticklabels(labels_)
    # axs[1].set_xticklabels(labels_)
    # axs[2].set_xticklabels(labels_)

    date_format = mdates.DateFormatter('%d/%b/%Y')
    x_lims = mdates.date2num(time_axis)

    annotation, missing_ids = create_annotation_matrix(df_raw, time_axis, day_before_famacha_test)

    im_a = axs[0].imshow(np.log(df_raw.iloc[:, :-4].values), cmap='gray', aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    plt.colorbar(im_a, ax=axs[0])

    im_e = axs[1].imshow(df_raw_e.iloc[:, :-4].values, cmap='gray', aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    plt.colorbar(im_e, ax=axs[1])

    im_m = axs[2].imshow(np.log(df_raw_m.iloc[:, :-4].values), cmap='gray', aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    plt.colorbar(im_m, ax=axs[2])

    axs[0].xaxis_date()
    axs[0].xaxis.set_major_formatter(date_format)
    axs[1].xaxis_date()
    axs[1].xaxis.set_major_formatter(date_format)
    axs[2].xaxis_date()
    axs[2].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    animal_ids_formatted_ent = df_raw["id"].values[::-1]
    axs[0].set_yticklabels(animal_ids_formatted_ent)
    axs[0].set_yticks(np.arange(len(animal_ids_formatted_ent)))
    axs[1].set_yticklabels(animal_ids_formatted_ent)
    axs[1].set_yticks(np.arange(len(animal_ids_formatted_ent)))
    axs[2].set_yticklabels(animal_ids_formatted_ent)
    axs[2].set_yticks(np.arange(len(animal_ids_formatted_ent)))

    for i in range(len(axs[0].get_yticklabels())):
        if "*" in str(animal_ids_formatted_ent[i]):
            axs[0].get_yticklabels()[i].set_color("tab:red")
            axs[1].get_yticklabels()[i].set_color("tab:red")
            axs[2].get_yticklabels()[i].set_color("tab:red")

    print("adding annotations...")
    activity_list_matrix = df_raw.iloc[:, :-4].values
    for i in range(activity_list_matrix.shape[0]):
        cpt = 0
        for j in range(activity_list_matrix.shape[1]):
            an = annotation[i, j]
            if an == "n" or an == "":
                cpt = 0
                continue
            cpt += 1
            color = "tab:gray"

            if an == "1To1":
                color = "tab:cyan"
            if an == "1To2":
                color = "tab:orange"
            if an == "2To2":
                color = "tab:blue"
            if an == "2To1":
                color = "tab:green"
            if an == "3To2":
                color = "tab:olive"
            #use ASCII 219 █ for text highlight instead of rectangle
            offset_y = 0.6
            offset_x = 0.9
            # ax.text(j-offset_x, activity_list_matrix.shape[0] - i - offset_y, "█", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')
            w = day_before_famacha_test
            lw = 2
            if cpt == 1:
                rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 1, fill=False, edgecolor=color, facecolor=None, lw=lw, alpha=0.8)
                axs[0].add_patch(rec)
                rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 1, fill=False, edgecolor=color,
                                facecolor=None, lw=lw, alpha=0.8)
                axs[1].add_patch(rec)
                rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 1, fill=False, edgecolor=color,
                                facecolor=None, lw=lw, alpha=0.8)
                axs[2].add_patch(rec)


    param_str = "sampling=%s day_before_famacha_test=%d" % (sampling, day_before_famacha_test)
    ntrans_with_samples = len(animal_ids_formatted_ent) - len(missing_ids)
    axs[0].set_title("Activity data sumed per %d day(s) %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (1, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))
    axs[1].set_title("Entropy data per %d day(s) %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (1, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))
    axs[2].set_title("Median data per %d day(s) %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (1, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))

    patch1 = mpatches.Patch(color='tab:cyan', label="1To1 "+str(DATASET_INFO["1To1"]))
    patch2 = mpatches.Patch(color='tab:orange', label="1To2 "+str(DATASET_INFO["1To2"]))
    patch3 = mpatches.Patch(color='tab:blue', label="2To2 "+str(DATASET_INFO["2To2"]))
    patch4 = mpatches.Patch(color='tab:green', label="2To1 "+str(DATASET_INFO["2To1"]))
    patch5 = mpatches.Patch(color='tab:olive', label="3To2 "+str(DATASET_INFO["3To2"]))
    axs[0].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)
    axs[1].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)
    axs[2].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)

    axs[0].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
    axs[1].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
    axs[2].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))

    axs[0].set_facecolor('pink')
    axs[1].set_facecolor('pink')
    axs[2].set_facecolor('pink')

    fig.tight_layout()

    filename = "dataset_heatmap_%s.png" % (farm_id)
    create_rec_dir(out_DIR)
    file_path = out_DIR +"/"+ filename.replace("=", "_")
    print(file_path)
    fig.savefig(file_path, bbox_inches='tight')
    fig.savefig(file_path.replace(".png", ".svg"))

    plt.show()

    print("done.")







