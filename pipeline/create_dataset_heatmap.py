import argparse
import glob
import json
import math
import sys
import matplotlib
#matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import scipy.stats
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from utils.Utils import anscombe


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
    if to_resample.dropna().size > 0:
        input_array = to_resample.fillna(2)
        input_array[input_array == 0] = 1
        e = scipy.stats.entropy(input_array)
    return e


def sum_(to_resample):
    s = np.nan
    if to_resample.dropna().size > 0:
        s = np.sum(to_resample.dropna())
    return s


def median_(to_resample):
    m = np.nan
    if to_resample.dropna().size > 0:
        m = np.nanmedian(to_resample)
    return m


def resample(df, animal_id, res=None):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(dict(first_sensor_value=sum_, first_sensor_value_gain=sum_, signal_strength=median_, battery_voltage=median_, xmin=sum_, xmax=sum_, ymin=sum_, ymax=sum_, zmin=sum_, zmax=sum_))
    # df_resampled = df.resample(res).agg(sum_)
    # df_resampled_entropy = df.resample(res).agg(dict(first_sensor_value=entropy_, signal_strength=entropy_, battery_voltage=entropy_, xmin=entropy_, xmax=entropy_, ymin=entropy_, ymax=entropy_, zmin=entropy_, zmax=entropy_))
    # df_resampled_median = df.resample(res).agg(dict(first_sensor_value=median_, signal_strength=median_, battery_voltage=median_, xmin=median_, xmax=median_, ymin=median_, ymax=median_, zmin=median_, zmax=median_))
    return df_resampled, df_resampled, df_resampled, res


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


def process_activity_data(file, i, nfiles, w, res, start, end):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")

    #w = 1440 * 3
    if w is None or w < 0:
        w = df_activity.shape[0]
    if end is None or end < 0:
        end = df_activity.shape[0]
    results = []
    cpt = 0

    # start = 0
    # end = df_activity.shape[0]
    # if start is not None and end is not None:
    #     w = 1440 * 3
    #     start = 540002
    #     # end = start + w*10
    #     end = df_activity.shape[0]

    for i in range(start, end, w):
        # print(animal_id, i, i+w)
        df_activity_w = df_activity.loc[i: i+w, :]
        # print(df_activity_w)
        # 411989 2015-11-04T02:29
        #159840
        if df_activity_w.shape[0] != w+1:
            continue
        entropy = scipy.stats.entropy(df_activity["first_sensor_value"].dropna())

        e_xmin = scipy.stats.entropy(df_activity["xmin"].dropna().abs())
        e_xmax = scipy.stats.entropy(df_activity["xmax"].dropna().abs())
        e_ymin = scipy.stats.entropy(df_activity["ymin"].dropna().abs())
        e_ymax = scipy.stats.entropy(df_activity["ymax"].dropna().abs())
        e_zmin = scipy.stats.entropy(df_activity["zmin"].dropna().abs())
        e_zmax = scipy.stats.entropy(df_activity["zmax"].dropna().abs())
        entropy_s2 = e_xmin + e_xmax + e_ymin + e_ymax + e_zmin + e_zmax

        if np.isnan(entropy_s2):
            entropy_s2 = 0


        if np.isnan(entropy):
            entropy = 0
        # entropy = entropy2(df_activity["first_sensor_value"].dropna().values)

        #add herd start and end to create missing empty bins of full time range
        # data = []
        # data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(start_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
        # df_activity = pd.concat([pd.DataFrame(data), df_activity], ignore_index=True)
        # data = []
        # data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(end_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
        # df_activity = pd.concat([df_activity, pd.DataFrame(data)], ignore_index=True)

        df_resampled_activity, df_resampled_entropy, df_resampled_median, resolution = resample(df_activity_w, animal_id, res=res)
        time = df_resampled_activity.index.values
        activity = df_resampled_activity.first_sensor_value.values
        activity_i = df_resampled_activity.first_sensor_value_gain.values
        activity_e = df_resampled_entropy.first_sensor_value.values
        activity_m = df_resampled_median.first_sensor_value.values

        merge_a = activity.tolist() + [entropy, entropy_s2, animal_id]
        merge_a_i = activity_i.tolist() + [entropy, entropy_s2, animal_id]
        merge_e = activity_e.tolist() + [entropy, entropy_s2, animal_id]
        merge_m = activity_m.tolist() + [entropy, entropy_s2, animal_id]
        merge_bat = df_resampled_activity.battery_voltage.values.tolist() + [entropy, entropy_s2, animal_id]
        merg_ss = df_resampled_activity.signal_strength.values.tolist() + [entropy, entropy_s2, animal_id]
        merge_xmin = df_resampled_activity.xmin.values.tolist() + [entropy, entropy_s2, animal_id]
        merge_xmax = df_resampled_activity.xmax.values.tolist() + [entropy, entropy_s2, animal_id]

        merge_ymin = df_resampled_activity.ymin.values.tolist() + [entropy, entropy_s2, animal_id]
        merge_ymax = df_resampled_activity.ymax.values.tolist() + [entropy, entropy_s2, animal_id]

        merge_zmin = df_resampled_activity.zmin.values.tolist() + [entropy, entropy_s2, animal_id]
        merge_zmax = df_resampled_activity.zmax.values.tolist() + [entropy, entropy_s2, animal_id]

        data = [animal_id, time, entropy, entropy_s2, merge_a, merge_a_i, merge_e, merge_m, merge_bat, merg_ss, merge_xmin, merge_xmax, merge_ymin, merge_ymax, merge_zmin, merge_zmax, resolution, cpt, str(i)+"_"+str(i+w)]
        # if cpt > 14:
        #     break
        results.append(data)
        cpt += 1

    return results


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
    print("create_annotation_matrix...")
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


def add_famacha_format_id(row, fam_data, s2):
    id = int(row["id"])
    if id in fam_data.keys():
        timestamps = fam_data[id]
        timestamps.sort(key=lambda x: x[0])
        row["famacha"] = timestamps
        if s2:
            row["id"] = str(row["id"]) + "  %06.2f" % row["entropy_s2"]
        else:
            row["id"] = str(row["id"]) + "  %06.2f" % row["entropy"]
    else:
        if s2:
            row["id"] = str(row["id"])[1:] + "*" + "  %06.2f" % row["entropy_s2"]
        else:
            row["id"] = str(row["id"])[1:] + "*" + "  %06.2f" % row["entropy"]
    return row


# def test():
#     host = host_subplot(111, axes_class=AA.Axes, figsize=(19.20, 10.80))
#     plt.subplots_adjust(right=0.75)
#
#     par1 = host.twinx()
#     par2 = host.twinx()
#     par3 = host.twinx()
#
#     offset = 60
#     new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#     par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
#                                         offset=(offset, 0))
#     par2.axis["right"].toggle(all=True)
#
#     new_fixed_axis3 = par3.get_grid_helper().new_fixed_axis
#     par3.axis["right"] = new_fixed_axis3(loc="right", axes=par3,
#                                         offset=(offset*2, 0))
#     par3.axis["right"].toggle(all=True)
#
#     host.set_xlabel("Time (days)")
#     host.set_ylabel("activity (first sensor)")
#     par1.set_ylabel("signal strenght (dB)")
#     par2.set_ylabel("battery level (V)")
#     par3.set_ylabel("acceleration sensor 2 (m.s^-1)")
#
#     host.plot(time_axis, activity, label="activity (first sensor)")
#     par1.plot(time_axis, ss, label="signal strenght (dB)")
#     par2.plot(time_axis, bat, label="battery level (V)")
#     par3.plot(time_axis, xmin, label="xmin")
#     par3.plot(time_axis, xmax, label="xmax")
#     par3.plot(time_axis, ymin, label="ymin")
#     par3.plot(time_axis, ymax, label="ymax")
#     par3.plot(time_axis, zmin, label="zmin")
#     par3.plot(time_axis, zmax, label="zmax")
#
#     host.legend()
#     plt.draw()
#     plt.show()


def export_tranponder_traces(row, rowss, rowbat, rowxmin, rowxmax, rowymin, rowymax, rowzmin, rowzmax, out_dir, farm_id, time_axis, i_c, i_t):
    print("export_tranponder_traces %d/%d..." % (i_c, i_t))
    export_dir = out_dir + "/transponder_export"
    create_rec_dir(export_dir)
    plt.clf()

    activity = row[:-5].values
    ss = rowss[:-5].values
    bat = rowbat[:-5].values
    xmin = rowxmin[:-5].values
    xmax = rowxmax[:-5].values
    ymin = rowymin[:-5].values
    ymax = rowymax[:-5].values
    zmin = rowzmin[:-5].values
    zmax = rowzmax[:-5].values

    entropy = row["entropy"]
    id = row["id"].split(" ")[0]
    # fig, ax = plt.subplots(figsize=(19.20, 10.80))
    # ax.plot(time_axis, activity, label="activity (first sensor)")
    # ax.plot(time_axis, ss, label="signal strenght (dB)")
    # ax.plot(time_axis, bat, label="battery level (V)")
    # ax.plot(time_axis, xmin, label="xmin")
    # ax.plot(time_axis, xmax, label="xmax")
    # ax.plot(time_axis, ymin, label="ymin")
    # ax.plot(time_axis, ymax, label="ymax")
    # ax.plot(time_axis, zmin, label="zmin")
    # ax.plot(time_axis, zmax, label="zmax")
    #
    # xlabel = "Time (days)"
    # ylabel = "Activity count"
    # ax.set(xlabel=xlabel, ylabel=ylabel)
    # ax.set_title("%s activity output of transponder %s entropy of entire trace=%.4f" % (farm_id, id, entropy))
    # #plt.show()
    # filename = "%s_%.4f_%s" % (farm_id, entropy, id)
    # filename = filename.replace(".", "_").replace("*", "") + ".png"
    # filepath = "%s/%s" % (export_dir, filename)
    # print('saving fig...')
    # fig.savefig(filepath)
    # print("saved!")
    # fig.clear()
    # plt.close(fig)
    plt.figure(figsize=(19.20, 10.80))
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    # par3 = host.twinx()

    offset = 60
    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="right", axes=par1,
                                        offset=(0, 0))
    par2.axis["right"].toggle(all=True)

    new_fixed_axis2 = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis2(loc="right", axes=par2,
                                        offset=(offset, 0))
    par2.axis["right"].toggle(all=True)

    # new_fixed_axis3 = par3.get_grid_helper().new_fixed_axis
    # par3.axis["right"] = new_fixed_axis3(loc="right", axes=par3,
    #                                     offset=(offset*2, 0))
    # par3.axis["right"].toggle(all=True)

    host.set_xlabel("Time (days)")
    host.set_ylabel("activity (first sensor)")
    par1.set_ylabel("signal strenght (dB)")
    par2.set_ylabel("magnitude sensor 2 sqrt(x^2+y^2+z^2)")
    # par3.set_ylabel("acceleration sensor 2 (m.s^-1)")

    host.plot(time_axis, activity, label="activity (first sensor)")
    par1.plot(time_axis, ss, label="signal strenght (dB)")
    # par2.plot(time_axis, bat, label="battery level (V)")

    amin = xmin * xmin + ymin * ymin + zmin * zmin
    s2min = np.sqrt(amin.tolist())
    amax = xmax * xmax + ymax * ymax + zmax * zmax
    s2max = np.sqrt(amax.tolist())

    par2.plot(time_axis, s2min, c='black', label="sensor 2 min magnitude", linestyle='-')
    par2.plot(time_axis, s2max, c='black', label="sensor 2 max magnitude", linestyle='--')

    host.legend()
    # plt.draw()
    # plt.show()
    filename = "%s_%.4f_%s" % (farm_id, entropy, id)
    filename = filename.replace(".", "_").replace("*", "") + ".png"
    filepath = "%s/%s" % (export_dir, filename)
    print('saving fig...')
    plt.savefig(filepath)
    print("saved!")


def add_famacha_format_id_todf(df_raw, header, famacha_data, s2=False):
    df_raw.columns = header
    df_raw["famacha"] = np.nan
    df_raw = df_raw.apply(add_famacha_format_id, axis=1, args=(famacha_data, s2))
    df_raw["possible"] = ['*' in x for x in df_raw["id"].values]
    if s2:
        df_raw = df_raw.sort_values(['possible', 'entropy_s2'], ascending=[True, False]).groupby('possible').head(df_raw.shape[0])
    else:
        df_raw = df_raw.sort_values(['possible', 'entropy'], ascending=[True, False]).groupby('possible').head(
            df_raw.shape[0])

    df_raw = df_raw.reset_index(drop=True)
    # df_raw = df_raw.sort_values(['entropy'], ascending=False, ignore_index=True)
    # print(df_raw)
    return df_raw


def compute_magnitude(a_xmin, a_ymin, a_zmin):
    magnitude = np.full(a_xmin.shape, np.nan)
    for i in range(a_xmin.shape[0]):
        for j in range(a_xmin.shape[1]):
            x = a_xmin[i, j]
            y = a_ymin[i, j]
            z = a_zmin[i, j]
            m = math.sqrt(x*x + y*y + z*z)
            magnitude[i, j] = m
    return magnitude


def create_heatmap(DATA, k, idx, itot, famacha_data, day_before_famacha_test, farm_id, DATASET_INFO, out_DIR, n_job):
    print("create_heatmap")
    print("progress create_heatmap %d/%d ..." % (idx, itot))
    # activity_list = []
    time_axis = None
    # animal_ids = []
    entropy_list = []
    entropy_s2_list = []
    raw = []
    raw_i = []
    raw_e = []
    raw_m = []
    raw_bat = []
    raw_ss = []
    raw_xmin = []
    raw_xmax = []
    raw_ymin = []
    raw_ymax = []
    raw_zmin = []
    raw_zmax = []
    resolution = ''
    wid = 0
    range_id = ''
    for item in DATA:
        # animal_ids.append(item[0])
        time_axis = item[k][1]
        entropy_list.append(item[k][2])
        entropy_s2_list.append(item[k][3])
        raw.append(item[k][4])
        raw_i.append(item[k][5])
        raw_e.append(item[k][6])
        raw_m.append(item[k][7])

        raw_bat.append(item[k][8])
        raw_ss.append(item[k][9])
        raw_xmin.append(item[k][10])
        raw_xmax.append(item[k][11])
        raw_ymin.append(item[k][12])
        raw_ymax.append(item[k][13])
        raw_zmin.append(item[k][14])
        raw_zmax.append(item[k][15])

        resolution = item[k][16]
        wid = item[k][17]
        range_id = item[k][18]

    df_raw = pd.DataFrame(raw, dtype=object)
    df_raw_i = pd.DataFrame(raw_i, dtype=object)
    # df_raw_e = pd.DataFrame(raw_e, dtype=object)
    # df_raw_m = pd.DataFrame(raw_m, dtype=object)

    df_raw_bat = pd.DataFrame(raw_bat, dtype=object)
    df_raw_ss = pd.DataFrame(raw_ss, dtype=object)

    df_raw_xmin = pd.DataFrame(raw_xmin, dtype=object)
    df_raw_xmax = pd.DataFrame(raw_xmax, dtype=object)

    df_raw_ymin = pd.DataFrame(raw_ymin, dtype=object)
    df_raw_ymax = pd.DataFrame(raw_ymax, dtype=object)

    df_raw_zmin = pd.DataFrame(raw_zmin, dtype=object)
    df_raw_zmax = pd.DataFrame(raw_zmax, dtype=object)

    header = [x for x in range(df_raw.shape[1])]
    header[-1] = "id"
    header[-2] = "entropy_s2"
    header[-3] = "entropy"

    print("add_famacha_format_id_todf... %d/%d ..." % (idx, itot))
    df_raw = add_famacha_format_id_todf(df_raw, header, famacha_data)
    df_raw_i = add_famacha_format_id_todf(df_raw_i, header, famacha_data)
    df_raw_ss = add_famacha_format_id_todf(df_raw_ss, header, famacha_data)
    df_raw_bat = add_famacha_format_id_todf(df_raw_bat, header, famacha_data)
    df_raw_xmin = add_famacha_format_id_todf(df_raw_xmin, header, famacha_data, s2=True)
    df_raw_xmax = add_famacha_format_id_todf(df_raw_xmax, header, famacha_data, s2=True)
    df_raw_ymin = add_famacha_format_id_todf(df_raw_ymin, header, famacha_data, s2=True)
    df_raw_ymax = add_famacha_format_id_todf(df_raw_ymax, header, famacha_data, s2=True)
    df_raw_zmin = add_famacha_format_id_todf(df_raw_zmin, header, famacha_data, s2=True)
    df_raw_zmax = add_famacha_format_id_todf(df_raw_zmax, header, famacha_data, s2=True)
    print("add_famacha_format_id_todf done %d/%d ..." % (idx, itot))

    # df_raw.columns = header
    # df_raw["famacha"] = np.nan
    # df_raw = df_raw.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    # df_raw["possible"] = ['*' in x for x in df_raw["id"].values]
    # df_raw = df_raw.sort_values(['possible', 'entropy'], ascending=[True, False]).groupby('possible').head(df_raw.shape[0])
    # df_raw = df_raw.reset_index(drop=True)
    # # df_raw = df_raw.sort_values(['entropy'], ascending=False, ignore_index=True)
    # print(df_raw)
    #
    # df_raw_e.columns = header
    # df_raw_e["famacha"] = np.nan
    # df_raw_e = df_raw_e.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    # df_raw_e["possible"] = ['*' in x for x in df_raw_e["id"].values]
    # df_raw_e = df_raw_e.sort_values(['possible', 'entropy'], ascending=[True, False]).groupby('possible').head(df_raw.shape[0])
    # # df_raw_e = df_raw_e.sort_values(['entropy'], ascending=False, ignore_index=True)
    # df_raw_e = df_raw_e.reset_index(drop=True)
    #
    # df_raw_m.columns = header
    # df_raw_m["famacha"] = np.nan
    # df_raw_m = df_raw_m.apply(add_famacha_format_id, axis=1, args=(famacha_data,))
    # df_raw_m["possible"] = ['*' in x for x in df_raw_m["id"].values]
    # df_raw_m = df_raw_m.sort_values(['possible', 'entropy'], ascending=[True, False]).groupby('possible').head(df_raw.shape[0])
    # df_raw_m = df_raw_m.reset_index(drop=True)

    df_raw = df_raw[df_raw["possible"] == False]
    df_raw_i = df_raw_i[df_raw_i["possible"] == False]
    # df_raw_e = df_raw_e[df_raw_e["possible"] == False]
    # df_raw_m = df_raw_m[df_raw_m["possible"] == False]
    df_raw_ss = df_raw_ss[df_raw_ss["possible"] == False]
    df_raw_bat = df_raw_bat[df_raw_bat["possible"] == False]
    df_raw_xmin = df_raw_xmin[df_raw_xmin["possible"] == False]
    df_raw_xmax = df_raw_xmax[df_raw_xmax["possible"] == False]
    df_raw_ymin = df_raw_ymin[df_raw_ymin["possible"] == False]
    df_raw_ymax = df_raw_ymax[df_raw_ymax["possible"] == False]
    df_raw_zmin = df_raw_zmin[df_raw_zmin["possible"] == False]
    df_raw_zmax = df_raw_zmax[df_raw_zmax["possible"] == False]
    print("ready for figure %d/%d ..." % (idx, itot))
    #########################################################################
    # print("exporting individual traces...")
    # pool = Pool(processes=n_job)
    # for (index, row), (indexss, rowss), (indexbat, rowbat),\
    #     (indexxmin, rowxmin), (indexxmax, rowxmax),\
    #     (indexymin, rowymin), (indexymax, rowymax),\
    #     (indexzmin, rowzmin), (indexzmax, rowzmax) in zip(df_raw.iterrows(), df_raw_ss.iterrows(), df_raw_bat.iterrows(),
    #                                                    df_raw_xmin.iterrows(), df_raw_xmax.iterrows(),
    #                                                    df_raw_ymin.iterrows(), df_raw_ymax.iterrows(),
    #                                                    df_raw_zmin.iterrows(), df_raw_zmax.iterrows()):
    #     export_tranponder_traces(row, rowss, rowbat, rowxmin, rowxmax, rowymin, rowymax, rowzmin,
    #                                                 rowzmax, out_DIR, farm_id, time_axis, index, df_raw.shape[0])
        # pool.apply_async(export_tranponder_traces, (row, rowss, rowbat, rowxmin, rowxmax, rowymin, rowymax, rowzmin,
        #                                             rowzmax, out_DIR, farm_id, time_axis, index, df_raw.shape[0], ))
    # pool.close()
    # pool.join()
    # pool.terminate()
    # for index, row in df_raw.iterrows():
    #     export_tranponder_traces(row, out_DIR, farm_id, time_axis, index, df_raw.shape[0])
    ##########################################################################

    n = 2
    h = (df_raw.shape[0] * 30 * n) / 100
    w = 36.20 * 2
    fig, axs = plt.subplots(n, figsize=(w, h))
    for p in range(n):
        axs[p].yaxis.set_label_position("right")
        axs[p].yaxis.tick_right()
        # axs[1].yaxis.tick_right()
        # axs[1].yaxis.set_label_position("right")
    # axs[2].yaxis.tick_right()
    # axs[2].yaxis.set_label_position("right")

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

    date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
    x_lims = mdates.date2num(time_axis)

    annotation, missing_ids = create_annotation_matrix(df_raw, time_axis, day_before_famacha_test)

    a = df_raw.iloc[:, :-5].values
    a_i = df_raw_i.iloc[:, :-5].values
    ss = df_raw_ss.iloc[:, :-5].values
    bat = df_raw_bat.iloc[:, :-5].values

    a_xmin = df_raw_xmin.iloc[:, :-5].values
    a_xmax = df_raw_xmax.iloc[:, :-5].values
    a_ymin = df_raw_ymin.iloc[:, :-5].values
    a_ymax = df_raw_ymax.iloc[:, :-5].values
    a_zmin = df_raw_zmin.iloc[:, :-5].values
    a_zmax = df_raw_zmax.iloc[:, :-5].values

    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([0 / 256, 0 / 256, 0 / 256, 1])
    newcolors[:1, :] = pink
    newcmp = ListedColormap(newcolors)

    # im_a_log = axs[0].imshow(np.log10(a, out=np.zeros_like(a), where=(a != 0)), cmap=newcmp, aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_a_log, ax=axs[0])

    # anscombe_m = np.vectorize(anscombe)
    # a_log_anscombe = anscombe_m(np.log(a, out=np.zeros_like(a), where=(a != 0)))

    im_a_log_anscomb = axs[0].imshow(a, cmap=newcmp, aspect='auto',
                                     interpolation="nearest",
                                     extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    plt.colorbar(im_a_log_anscomb, ax=axs[0])
    axs[0].xaxis_date()

    im_a_i = axs[1].imshow(a_i, cmap=newcmp, aspect='auto',
                                     interpolation="nearest",
                                     extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    plt.colorbar(im_a_i, ax=axs[1])
    axs[1].xaxis_date()

    # e = df_raw_e.iloc[:, :-4].values
    # im_e = axs[1].imshow(e, aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_e, ax=axs[1])
    #
    # m = df_raw_m.iloc[:, :-4].values
    # im_m = axs[2].imshow(m, aspect='auto', interpolation="nearest", extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_m, ax=axs[2])

    # im_ss = axs[1].imshow(ss, cmap=newcmp, aspect='auto',
    #                          interpolation="nearest",
    #                          extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_ss, ax=axs[1])
    #
    # im_bat = axs[2].imshow(bat, cmap=newcmp, aspect='auto',
    #                          interpolation="nearest",
    #                          extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_bat, ax=axs[2])
    #
    # # im_xmin = axs[3].imshow(a_xmin, cmap=newcmp, aspect='auto',
    # #                          interpolation="nearest",
    # #                          extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # plt.colorbar(im_xmin, ax=axs[3])
    # #
    # # im_xmax = axs[4].imshow(a_xmax, cmap=newcmp, aspect='auto',
    # #                          interpolation="nearest",
    # #                          extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # # plt.colorbar(im_xmax, ax=axs[4])
    # #
    # # ###
    # # im_ymin = axs[5].imshow(a_ymin, cmap=newcmp, aspect='auto',
    # #                         interpolation="nearest",
    # #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # plt.colorbar(im_ymin, ax=axs[5])
    # #
    # # im_ymax = axs[6].imshow(a_ymax, cmap=newcmp, aspect='auto',
    # #                         interpolation="nearest",
    # #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # plt.colorbar(im_ymax, ax=axs[6])
    # #
    # # ###
    # # im_zmin = axs[7].imshow(a_zmin, cmap=newcmp, aspect='auto',
    # #                         interpolation="nearest",
    # #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # # plt.colorbar(im_zmin, ax=axs[7])
    # #
    # # im_zmax = axs[8].imshow(a_zmax, cmap=newcmp, aspect='auto',
    # #                         interpolation="nearest",
    # #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # # plt.colorbar(im_zmax, ax=axs[8])
    #
    # ##magnitude
    # min_magnitude = compute_magnitude(a_xmin, a_ymin, a_zmin)
    # mmin_log_anscombe = anscombe_m(np.log(min_magnitude, out=np.zeros_like(min_magnitude), where=(min_magnitude != 0)))
    # im_min_magnitude = axs[3].imshow(min_magnitude, cmap=newcmp, aspect='auto',
    #                         interpolation="nearest",
    #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_min_magnitude, ax=axs[3])
    #
    # max_magnitude = compute_magnitude(a_xmax, a_ymax, a_zmax)
    # mmax_log_anscombe = anscombe_m(np.log(max_magnitude, out=np.zeros_like(max_magnitude), where=(max_magnitude != 0)))
    # im_max_magnitude = axs[4].imshow(max_magnitude, cmap=newcmp, aspect='auto',
    #                         interpolation="nearest",
    #                         extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
    # plt.colorbar(im_max_magnitude, ax=axs[4])

    if resolution == "1T":
        for p in range(n):
            axs[p].xaxis_date()
            axs[p].xaxis.set_major_formatter(date_format)
            axs[p].xaxis.set_major_locator(mdates.MinuteLocator(interval=60))

        # axs[0].xaxis_date()
        # axs[0].xaxis.set_major_formatter(date_format)
        # axs[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        # axs[1].xaxis_date()
        # axs[1].xaxis.set_major_formatter(date_format)
        # axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    elif resolution == "1D":
        for p in range(n):
            # axs[p].xaxis_date()
            axs[p].xaxis.set_major_formatter(date_format)
            axs[p].xaxis.set_major_locator(mdates.DayLocator(interval=7))

        # axs[0].xaxis_date()
        # axs[0].xaxis.set_major_formatter(date_format)
        # axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        #
        # axs[1].xaxis_date()
        # axs[1].xaxis.set_major_formatter(date_format)
        # axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    else:
        for p in range(n):
            # axs[p].xaxis_date()
            axs[p].xaxis.set_major_formatter(date_format)
            axs[p].xaxis.set_major_locator(mdates.DayLocator(interval=7))

        # axs[0].xaxis_date()
        # axs[0].xaxis.set_major_formatter(date_format)
        # axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        # axs[1].xaxis_date()
        # axs[1].xaxis.set_major_formatter(date_format)
        # axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))

    # axs[2].xaxis_date()
    # axs[2].xaxis.set_major_formatter(date_format)
    # axs[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    fig.autofmt_xdate()

    animal_ids_formatted_ent = df_raw["id"].values[::-1]
    animal_ids_formatted_ent_s2 = df_raw_xmin["id"].values[::-1]
    # axs[0].set_yticklabels(animal_ids_formatted_ent)
    # axs[0].set_yticks(np.arange(len(animal_ids_formatted_ent)))
    # axs[1].set_yticklabels(animal_ids_formatted_ent)
    # axs[1].set_yticks(np.arange(len(animal_ids_formatted_ent)))
    for p in range(n):
        axs[p].set_yticklabels(animal_ids_formatted_ent)
        axs[p].set_yticks(np.arange(len(animal_ids_formatted_ent)))
        if p > 2:
            axs[p].set_yticklabels(animal_ids_formatted_ent_s2)
            axs[p].set_yticks(np.arange(len(animal_ids_formatted_ent_s2)))

    # axs[2].set_yticklabels(animal_ids_formatted_ent)
    # axs[2].set_yticks(np.arange(len(animal_ids_formatted_ent)))

    for i in range(len(axs[0].get_yticklabels())):
        if "*" in str(animal_ids_formatted_ent[i]):
            # axs[0].get_yticklabels()[i].set_color("tab:red")
            # axs[1].get_yticklabels()[i].set_color("tab:red")
            # axs[2].get_yticklabels()[i].set_color("tab:red")
            for p in range(n):
                axs[p].get_yticklabels()[i].set_color("tab:red")

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
            color = "lightgrey"

            if an == "1To1":
                color = "white"
            if an == "1To2":
                color = "red"
            if an == "2To2":
                color = "blue"
            if an == "2To1":
                color = "orange"
            if an == "3To2":
                color = "lawngreen"
            #use ASCII 219 for text highlight instead of rectangle
            offset_y = 0.6
            offset_x = 0.9
            # ax.text(j-offset_x, activity_list_matrix.shape[0] - i - offset_y, "x", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')
            w = day_before_famacha_test
            lw = 1.4
            if cpt == 1:
                # rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 0.85, fill=False, edgecolor=color, facecolor=None, lw=lw, alpha=1)
                # axs[0].add_patch(rec)
                #
                # rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 0.85, fill=False, edgecolor=color, facecolor=None, lw=lw, alpha=1)
                # axs[1].add_patch(rec)

                for p in range(n):
                    rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 0.85, fill=False,
                                    edgecolor=color, facecolor=None, lw=lw, alpha=1)
                    axs[p].add_patch(rec)
                # rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 1, fill=False, edgecolor=color,
                #                 facecolor=None, lw=lw, alpha=0.8)
                # axs[2].add_patch(rec)


    param_str = "sampling=%s day_before_famacha_test=%d" % (resolution, day_before_famacha_test)
    ntrans_with_samples = len(animal_ids_formatted_ent) - len(missing_ids)
    axs[0].set_title("Activity raw data per %s  %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (resolution, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))
    axs[1].set_title("Imputed Activity raw data per %s  %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (resolution, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))
    # axs[2].set_title("Median data per %s  %s herd and dataset samples location\n%s\n%s\n*no famacha data corresponding animal id size=%d/%d\ntransponder traces with fam samples=%d" % (resolution, farm_id, breaklineinsert(str(DATASET_INFO)), param_str, len(missing_ids), len(animal_ids_formatted_ent), ntrans_with_samples))
    # axs[1].set_title("signal strenght")
    # axs[2].set_title("battery voltage")
    # # axs[3].set_title("accelerometer x min axis")
    # # axs[4].set_title("accelerometer x max axis")
    # # axs[5].set_title("accelerometer y min axis")
    # # axs[6].set_title("accelerometer y max axis")
    # # axs[7].set_title("accelerometer z min axis")
    # # axs[8].set_title("accelerometer z max axis")
    # axs[3].set_title("(log) Magnitute min")
    # axs[4].set_title("(log) Magnitude max")


    patch1 = mpatches.Patch(color='white', label="1To1 "+str(DATASET_INFO["1To1"]))
    patch2 = mpatches.Patch(color='red', label="1To2 "+str(DATASET_INFO["1To2"]))
    patch3 = mpatches.Patch(color='blue', label="2To2 "+str(DATASET_INFO["2To2"]))
    patch4 = mpatches.Patch(color='orange', label="2To1 "+str(DATASET_INFO["2To1"]))
    patch5 = mpatches.Patch(color='lawngreen', label="3To2 "+str(DATASET_INFO["3To2"]))

    for p in range(n):
        axs[p].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.5)
        # axs[1].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.5)
        # axs[2].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)

        axs[p].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
        # axs[1].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
        # axs[2].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))

        axs[p].set_facecolor('pink')
        # axs[1].set_facecolor('pink')
        # axs[2].set_facecolor('pink')

    fig.tight_layout()

    filename = "%s_dataset_heatmap_%s_%s_crop_color.png" % (range_id, farm_id, resolution)
    create_rec_dir(out_DIR)
    file_path = out_DIR +"/"+ filename.replace("=", "_")
    print("saving figure ", file_path)
    fig.savefig(file_path, bbox_inches='tight')
    print("saved ", filename)
    # fig.savefig(file_path.replace(".png", ".svg"))
    # plt.interactive(True)
    # plt.show()


def main(args):
    try:
        dataset_file_path = glob.glob(args.dataset_dir + "/*.csv")[0].replace("\\", '/')
    except IndexError as e:
        print(e)
        print("could not find dataset (.csv) file in %s" % args.dataset_dir)
    info_files = glob.glob(args.dataset_dir+"/*.json")
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
    # pool = Pool(processes=args.n_job)
    # results_dates = []
    # for i, file in enumerate(files):
    #     if 'median' in file or 'mean' in file:
    #         continue
    #     results_dates.append(pool.apply_async(get_start_end_date, (file, i, len(files))))
    # pool.close()
    # pool.join()
    # pool.terminate()
    # start_time = None
    # end_time = None
    # for res in results_dates:
    #     s = res.get()[0]
    #     e = res.get()[1]
    #     if start_time is None or s < start_time:
    #         start_time = s
    #     if end_time is None or e > end_time:
    #         end_time = e
    # start_time = np.datetime64(str(start_time).split('T')[0])
    # end_time = np.datetime64(str(end_time).split('T')[0])
    # end_time = end_time + np.timedelta64(1, 'D')
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
        results.append(pool.apply_async(process_activity_data, (file, i, len(files), args.w, args.res, args.start, args.end)))
    pool.close()
    pool.join()
    pool.terminate()

    DATA = []
    for res in results:
        DATA.append(res.get())

    ######################################################
    print("starting second pool.")
    # MAX_THREADC = 6
    MAX_THREADC = args.n_job
    njob = MAX_THREADC if args.n_job >= MAX_THREADC else args.n_job
    # pool2 = Pool(processes=njob)
    print("with njob=%d" % njob)
    r_ = list(range(len(DATA[0])))

    for i, k in enumerate(r_):
        print("feeding pool", i)
        # print(k, i, len(DATA[0]), day_before_famacha_test, farm_id, out_DIR)
        create_heatmap(DATA, k, i, len(DATA[0]), famacha_data, day_before_famacha_test, farm_id, DATASET_INFO, out_DIR, args.n_job)
    #     pool2.apply_async(create_heatmap, (DATA, k, i, len(DATA[0]), famacha_data, day_before_famacha_test, farm_id, DATASET_INFO, out_DIR, args.n_job))
    # pool2.close()
    # pool2.join()
    # pool2.terminate()

    print("done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create heatmap of the heard with sample overlay.')
    parser.add_argument('output',
                        help='Directory path of heatmap output. (Directory will be created if does not exist)')
    parser.add_argument('activity_dir', help='Parent directory of the activity data.')
    parser.add_argument('dataset_dir', help='Path of the directory containing dataset .csv and class info .txt.')
    parser.add_argument('--w', type=int, default=1440 * 3, help='Size of slicing window in minutes. (pass negative value for entire signal trace)')
    parser.add_argument('--res', type=str, default='1T', help='Sampling resolution.')
    parser.add_argument('--start', type=int, default=0, help='start time in minute.')
    parser.add_argument('--end', type=int, default=-1, help='end time in minute.')
    parser.add_argument('--n_job', type=int, default=1, help='Number of thread to use.')
    args = parser.parse_args()

    print("Argument values:")
    print("output=", args.output)
    print("activity_dir=", args.activity_dir)
    print("dataset_dir=", args.dataset_dir)
    print("w=", args.w)
    print("res=", args.res)
    print("start=", args.start)
    print("end=", args.end)
    print("njob=", args.n_job)

    main(args)







