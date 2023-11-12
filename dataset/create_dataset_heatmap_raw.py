import glob
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import numpy as np
import pandas as pd
import scipy.stats
import typer
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import host_subplot
from tqdm import tqdm
import matplotlib.ticker as ticker

from utils.Utils import anscombe

colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
colorst = [colormap(i) for i in np.linspace(0, 0.9, 23)]
COLOR_MAP = {
    "4To2": colorst[0],
    "1To2": colorst[1],
    "2To3": colorst[2],
    "1To1": colorst[3],
    "2To5": colorst[4],
    "3To3": colorst[5],
    "2To1": colorst[6],
    "4To1": colorst[7],
    "5To2": colorst[8],
    "4To3": colorst[9],
    "4To4": colorst[10],
    "3To4": colorst[11],
    "2To4": colorst[12],
    "5To3": colorst[13],
    "1To4": colorst[14],
    "3To5": colorst[15],
    "3To1": colorst[16],
    "3To2": colorst[17],
    "2To2": colorst[18],
    "4To5": colorst[19],
    "1To3": colorst[20],
    "1To5": colorst[21],
    "5To1": colorst[22],
}


def breaklineinsert_(str):
    midPoint = len(str) // 2
    return str[:midPoint] + "\n" + str[midPoint:]


def breaklineinsert(str):
    midPoint = len(str) // 2
    return breaklineinsert_(str[:midPoint]) + "\n" + breaklineinsert_(str[midPoint:])


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
    filename = Path(dataset_filepath).name
    split = filename.split("_")
    farm_id = split[1] + "_" + split[2]
    sampling = split[5]
    day_before_famacha_test = int(split[4])
    return farm_id, sampling, day_before_famacha_test


def parse_animal_id(file):
    animal_id = int(Path(file).stem)
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


def resample(col, df, animal_id, res=None):
    df.index = pd.to_datetime(df.date_str)
    df_resampled = df.resample(res).agg(
        {col: sum_, "date_str": "first"}
        # dict(
        #     first_sensor_value=sum_,
        #     first_sensor_value_gain=sum_,
        #     signal_strength=median_,
        #     battery_voltage=median_,
        #     xmin=sum_,
        #     xmax=sum_,
        #     ymin=sum_,
        #     ymax=sum_,
        #     zmin=sum_,
        #     zmax=sum_,
        # )
    )
    # df_resampled = df.resample(res).agg(sum_)
    # df_resampled_entropy = df.resample(res).agg(dict(first_sensor_value=entropy_, signal_strength=entropy_, battery_voltage=entropy_, xmin=entropy_, xmax=entropy_, ymin=entropy_, ymax=entropy_, zmin=entropy_, zmax=entropy_))
    # df_resampled_median = df.resample(res).agg(dict(first_sensor_value=median_, signal_strength=median_, battery_voltage=median_, xmin=median_, xmax=median_, ymin=median_, ymax=median_, zmin=median_, zmax=median_))
    return df_resampled


def entropy2(labels, base=None):
    """Computes entropy of label distribution."""
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.0
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def process_activity_data(activity_colummn, file, i, nfiles, w, res, start, end, zoom):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(str(file))
    df_activity = pd.read_csv(file, sep=",")

    df_activity["xmin"] = 0
    df_activity["ymin"] = 0
    df_activity["zmin"] = 0
    df_activity["xmax"] = 0
    df_activity["ymax"] = 0
    df_activity["zmax"] = 0
    # df_activity["signal_strength"] = 0
    df_activity["battery_voltage"] = 0

    # w = 1440 * 3
    # if w is None or w < 0:
    #     w = df_activity.shape[0]
    # if end is None or end < 0:
    #     end = df_activity.shape[0]
    results = []
    cpt = 0

    # start = 0
    # end = df_activity.shape[0]
    # if start is not None and end is not None:
    #     w = 1440 * 3
    #     start = 540002
    #     # end = start + w*10
    #     end = df_activity.shape[0]

    # for i in range(start, end, w):
    # print(animal_id, i, i+w)
    df_activity_w = df_activity.loc[:, :]
    if zoom:
        df_activity_w = df_activity.loc[540002: 540002 +  1440 * 1, :]

    # print(df_activity_w)
    # 411989 2015-11-04T02:29
    # 159840
    # if df_activity_w.shape[0] - w > 1:
    #     continue
    entropy = scipy.stats.entropy(np.histogram(df_activity["first_sensor_value"].dropna(), density=True)[0])

    # e_xmin = scipy.stats.entropy(df_activity["xmin"].dropna().abs())
    # e_xmax = scipy.stats.entropy(df_activity["xmax"].dropna().abs())
    # e_ymin = scipy.stats.entropy(df_activity["ymin"].dropna().abs())
    # e_ymax = scipy.stats.entropy(df_activity["ymax"].dropna().abs())
    # e_zmin = scipy.stats.entropy(df_activity["zmin"].dropna().abs())
    # e_zmax = scipy.stats.entropy(df_activity["zmax"].dropna().abs())
    # entropy_s2 = e_xmin + e_xmax + e_ymin + e_ymax + e_zmin + e_zmax

    # if np.isnan(entropy_s2):
    #     entropy_s2 = 0

    if np.isnan(entropy):
        entropy = 0
    # entropy = entropy2(df_activity["first_sensor_value"].dropna().values)

    # add herd start and end to create missing empty bins of full time range
    # data = []
    # data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(start_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    # df_activity = pd.concat([pd.DataFrame(data), df_activity], ignore_index=True)
    # data = []
    # data.insert(0, {'timestamp': np.nan, 'date_str': pd.to_datetime(str(end_time)).strftime('%Y-%m-%dT%H:%M'), 'first_sensor_value': np.nan})
    # df_activity = pd.concat([df_activity, pd.DataFrame(data)], ignore_index=True)

    df_activity_w = resample(activity_colummn, df_activity_w, animal_id, res=res)
    df_activity_w.index = pd.to_datetime(df_activity_w.date_str)
    time = df_activity_w.index.values
    activity = df_activity_w[activity_colummn].values
    # activity_i = df_activity_w.first_sensor_value_gain.values
    # activity_e = df_activity_w.first_sensor_value.values
    # activity_m = df_activity_w.first_sensor_value.values

    merge_a = activity.tolist() + [entropy, 0, animal_id]
    # merge_a_i = activity_i.tolist() + [entropy, 0, animal_id]
    # merge_e = activity_e.tolist() + [entropy, entropy_s2, animal_id]
    # merge_m = activity_m.tolist() + [entropy, entropy_s2, animal_id]
    # merge_bat = df_resampled_activity.battery_voltage.values.tolist() + [entropy, entropy_s2, animal_id]
    # merg_ss = df_resampled_activity.signal_strength.values.tolist() + [entropy, entropy_s2, animal_id]
    # merge_xmin = df_resampled_activity.xmin.values.tolist() + [entropy, entropy_s2, animal_id]
    # merge_xmax = df_resampled_activity.xmax.values.tolist() + [entropy, entropy_s2, animal_id]
    #
    # merge_ymin = df_resampled_activity.ymin.values.tolist() + [entropy, entropy_s2, animal_id]
    # merge_ymax = df_resampled_activity.ymax.values.tolist() + [entropy, entropy_s2, animal_id]
    #
    # merge_zmin = df_resampled_activity.zmin.values.tolist() + [entropy, entropy_s2, animal_id]
    # merge_zmax = df_resampled_activity.zmax.values.tolist() + [entropy, entropy_s2, animal_id]
    (
        merge_bat,
        merg_ss,
        merge_xmin,
        merge_xmax,
        merge_ymin,
        merge_ymax,
        merge_zmin,
        merge_zmax,
        merge_e,
        merge_m,
    ) = ([], [], [], [], [], [], [], [], [], [])

    data = [
        animal_id,
        time,
        entropy,
        0,
        merge_a,
        merge_a,
        merge_e,
        merge_m,
        merge_bat,
        merg_ss,
        merge_xmin,
        merge_xmax,
        merge_ymin,
        merge_ymax,
        merge_zmin,
        merge_zmax,
        res,
        cpt,
        str(i) + "_" + str(i + w),
    ]
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
    data_frame = pd.read_csv(file, sep=",", header=None)
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    hearder[-4] = "label"
    hearder[-3] = "serial"
    hearder[-2] = "miss_rate"
    hearder[-1] = "dtf1"
    data_frame.columns = hearder
    return data_frame


def create_annotation_matrix(df, time_axis, window_size):
    print("create_annotation_matrix...")
    # annotation = np.random.rand(activity_list_matrix.shape[0], activity_list_matrix.shape[1])
    # annotation = np.ones(df.iloc[:, :-4].shape) * -1
    annotation = np.empty(df.iloc[:, :-4].shape, dtype="<U10")
    missing_ids = []

    cpt = 0
    for i in tqdm(range(annotation.shape[0])):
        a_id = df.iloc[i, :]["id"]
        data_fam = np.array(df.iloc[i, :]["famacha"])
        if len(data_fam.shape) == 0:
            print(
                "animal id %s in the activity data (csv files) do not exist in the famacha data."
                % a_id
            )
            annotation[i, :] = "n"
            missing_ids.append(a_id)
            continue

        for f_data in data_fam:
            for j in range(time_axis.shape[0]):
                date_f = f_data[0].to_datetime64()
                target = f_data[1]
                days = int(
                    (date_f - time_axis[j]).astype("timedelta64[D]")
                    / np.timedelta64(1, "D")
                )
                if days == 0:
                    annotation[i, j - window_size : j] = target
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


def export_tranponder_traces(
    row,
    rowss,
    rowbat,
    rowxmin,
    rowxmax,
    rowymin,
    rowymax,
    rowzmin,
    rowzmax,
    out_dir,
    farm_id,
    time_axis,
    i_c,
    i_t,
):
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
    par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
    par2.axis["right"].toggle(all=True)

    new_fixed_axis2 = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis2(loc="right", axes=par2, offset=(offset, 0))
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

    par2.plot(
        time_axis, s2min, c="black", label="sensor 2 min magnitude", linestyle="-"
    )
    par2.plot(
        time_axis, s2max, c="black", label="sensor 2 max magnitude", linestyle="--"
    )

    host.legend()
    # plt.draw()
    # plt.show()
    filename = "%s_%.4f_%s" % (farm_id, entropy, id)
    filename = filename.replace(".", "_").replace("*", "") + ".png"
    filepath = "%s/%s" % (export_dir, filename)
    print("saving fig...")
    plt.savefig(filepath)
    print("saved!")


def add_famacha_format_id_todf(df_raw, header, famacha_data, s2=False):
    df_raw.columns = header
    df_raw["famacha"] = np.nan
    df_raw = df_raw.apply(add_famacha_format_id, axis=1, args=(famacha_data, s2))
    df_raw["possible"] = ["*" in x for x in df_raw["id"].values]
    if s2:
        df_raw = (
            df_raw.sort_values(["possible", "entropy_s2"], ascending=[True, False])
            .groupby("possible")
            .head(df_raw.shape[0])
        )
    else:
        df_raw = (
            df_raw.sort_values(["possible", "entropy"], ascending=[True, False])
            .groupby("possible")
            .head(df_raw.shape[0])
        )

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
            m = math.sqrt(x * x + y * y + z * z)
            magnitude[i, j] = m
    return magnitude


def create_heatmap(
    display_famacha,
    activity_col,
    zoom,
    no_filter,
    filename,
    DATA,
    k,
    idx,
    itot,
    famacha_data,
    day_before_famacha_test,
    farm_id,
    DATASET_INFO,
    out_dir,
    n_job,
    w,
    h
):
    print("create_heatmap")
    print(f"progress create_heatmap {idx}/{itot} ...")
    # activity_list = []
    time_axis = None
    # animal_ids = []
    entropy_list = []
    entropy_s2_list = []
    raw = []
    resolution = ""
    wid = 0
    range_id = ""
    for item in DATA:
        # animal_ids.append(item[0])
        time_axis = item[k][1]
        entropy_list.append(item[k][2])
        entropy_s2_list.append(item[k][3])
        raw.append(item[k][4])
        resolution = item[k][16]
        wid = item[k][17]
        range_id = item[k][18]

    df_raw = pd.DataFrame(raw, dtype=object)
    df_raw[0][0] = 0
    print(df_raw)
    c = np.nansum(df_raw.iloc[:, :-4].values.astype(float))

    header = [x for x in range(df_raw.shape[1])]
    header[-1] = "id"
    header[-2] = "entropy_s2"
    header[-3] = "entropy"

    if df_raw.iloc[:, -1].values[-1] == 99999999999:
        df_raw = df_raw[:-1]

    print(f"add_famacha_format_id_todf... {idx}/{itot} ...")
    df_raw = add_famacha_format_id_todf(df_raw, header, famacha_data)
    print(f"add_famacha_format_id_todf done {idx}/{itot} ...")

    if no_filter:
        df_raw["possible"] = False

    if c != 0:
        df_raw = df_raw[df_raw["possible"] == False]

    print(f"ready for figure {idx}/{itot} ...")

    annotation, missing_ids = create_annotation_matrix(
        df_raw, time_axis, day_before_famacha_test
    )
    print("ANNOT", np.unique(annotation.flatten()).tolist())
    # f_id = [list(x)[1] for x in list(map(set, annotation))]
    f_id = []
    for x in list(map(set, annotation)):
        if len(x) != 2:
            f_id.append("NaN")
            continue
        f_id.append(list(x)[1])

    a = df_raw.iloc[:, :-5].values

    a = anscombe(np.abs(a))

    if "signal" in activity_col.lower():
        a = np.log(a)

    viridis = cm.get_cmap("viridis", 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([0 / 229, 0 / 236, 0 / 246, 1])
    newcolors[:1, :] = pink
    newcmp = ListedColormap(newcolors)

    date_format = mdates.DateFormatter("%H:00")
    x_lims = mdates.date2num(time_axis)

    # height = 13 * df_raw.shape[0]/100
    # figsize = (20.20, height)
    # if farm_id == "cedara":
    #     figsize = (20.20, height)

    figsize = (w, h)
    # figsize = (7.2, 9.8)
    # if farm_id == "cedara":
    #     figsize = (7.2, 12.8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    im_a_log_anscomb = ax.imshow(
        a,
        cmap=newcmp,
        aspect="auto",
        interpolation="nearest",
        extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]],
    )
    # plt.colorbar(im_a_log_anscomb, ax=[ax], location='left')
    # fig.colorbar(im_a_log_anscomb, cax=ax)
    if not no_filter:
        cb = plt.colorbar(im_a_log_anscomb, ax=[ax], location="left", pad=0.01)

    ax.xaxis_date()

    if resolution == "1T":
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60*3))

    elif resolution == "1D":
        # axs[p].xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    else:
        # axs[p].xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))

    fig.autofmt_xdate(rotation=90)
    if farm_id == "cedara":
        ax.axvline(pd.Timestamp("2013-02-14"), color="white", linestyle="--", lw=3)

    # every_nth = 2
    # if farm_id == "cedara":
    #     every_nth = 6
    # for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if n % every_nth == 0:
    #         label.set_visible(False)

    animal_ids_formatted_ent = df_raw["id"].values[::-1]
    animal_ids_formatted_ent = [x.split(" ")[0] for x in animal_ids_formatted_ent]
    # animal_ids_formatted_ent = np.array(
    #     [x[1] + " " + x[0] for x in zip(f_id, animal_ids_formatted_ent)]
    # )
    # animal_ids_formatted_ent_s2 = df_raw_xmin["id"].values[::-1]

    #ax.set_yticklabels(animal_ids_formatted_ent)
    # ax.set_yticklabels(np.arange(len(animal_ids_formatted_ent)))
    # ax.set_yticks(np.arange(len(animal_ids_formatted_ent)))
    # if p > 2:
    #     axs[p].set_yticklabels(animal_ids_formatted_ent_s2)
    #     axs[p].set_yticks(np.arange(len(animal_ids_formatted_ent_s2)))

    for i in range(len(ax.get_yticklabels())):
        if "*" in str(animal_ids_formatted_ent[i]):
            ax.get_yticklabels()[i].set_color("tab:red")

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
            color = COLOR_MAP[an]
            #
            # if an == "1To1":
            #     color = "white"
            # if an == "1To2":
            #     color = "red"
            # if an == "2To2":
            #     color = "blue"
            # if an == "2To1":
            #     color = "orange"
            # if an == "3To2":
            #     color = "lawngreen"
            # use ASCII 219 for text highlight instead of rectangle
            offset_y = 0.6
            offset_x = 0.9
            # ax.text(j-offset_x, activity_list_matrix.shape[0] - i - offset_y, "x", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')
            w = day_before_famacha_test
            lw = 1.4
            if cpt == 1:
                rec = Rectangle(
                    (x_lims[j] - 7, activity_list_matrix.shape[0] - i - 1),
                    w,
                    0.85,
                    fill=False,
                    edgecolor=color,
                    facecolor=None,
                    lw=lw,
                    alpha=1,
                )
                if display_famacha:
                    ax.add_patch(rec)

    param_str = (
        f"sampling={resolution} day_before_famacha_test={day_before_famacha_test}"
    )
    ntrans_with_samples = len(animal_ids_formatted_ent) - len(missing_ids)

    title1 = f"{farm_id} herd {activity_col} data (t=anscombe bin={resolution})"
    # ax.set_title(title1)

    title2 = f"Imputed Activity data per {resolution}  {farm_id} herd and dataset samples location\n{breaklineinsert(str(DATASET_INFO))}\n{param_str}\n*no famacha data corresponding animal id size={len(missing_ids)}/{len(animal_ids_formatted_ent)}\ntransponder traces with fam samples={ntrans_with_samples}"

    title3 = f"(imputed only) Imputed Activity raw data per {resolution}  {farm_id} herd and dataset samples location\n{breaklineinsert(str(DATASET_INFO))}\n{param_str}\n*no famacha data corresponding animal id size={len(missing_ids)}/{len(animal_ids_formatted_ent)}\ntransponder traces with fam samples={ntrans_with_samples}"
    patches = []
    if display_famacha:
        for k in DATASET_INFO.keys():
            if k == "total":
                continue
            patches.append(
                mpatches.Patch(color=COLOR_MAP[k], label=f"{k} " + str(DATASET_INFO[k]))
            )

    # patch1 = mpatches.Patch(
    #     color="lightgray", edgecolor="black", label="1To1 " + str(DATASET_INFO["1To1"])
    # )
    # patch2 = mpatches.Patch(color="red", label="1To2 " + str(DATASET_INFO["1To2"]))
    # patch3 = mpatches.Patch(color="blue", label="2To2 " + str(DATASET_INFO["2To2"]))
    # patch4 = mpatches.Patch(color="orange", label="2To1 " + str(DATASET_INFO["2To1"]))
    # patch5 = mpatches.Patch(
    #     color="lawngreen", label="3To2 " + str(DATASET_INFO["3To2"])
    # )

    y = 1.18
    if "cedara" in farm_id:
        y = 1.28

    if display_famacha:
        legend = ax.legend(
            title=title1.title(),
            frameon=False,
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            fancybox=True,
            framealpha=0.5,
            ncol=4,
        )
    else:
        ax.set_title(title1.title())

    # frame = legend.get_frame()  # sets up for color, edge, and transparency
    # frame.set_facecolor('#b4aeae')  # color of legend

    # axs[1].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.5)
    # axs[2].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)

    # ax.yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
    # axs[1].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
    # axs[2].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))

    # every_nth = 2
    # if farm_id == "cedara":
    #     every_nth = 15
    # for n, label in enumerate(ax.yaxis.get_ticklabels()):
    #     if n % every_nth == 0:
    #         label.set_visible(False)
    tick_spacing = 2
    if farm_id == "cedara":
        tick_spacing = 6
    # ax.set_yticks(ax.get_yticks()[::tick_spacing])
    # for i, label in enumerate(ax.get_yticklabels()):
    #     if i % tick_spacing != 0:
    #         label.set_visible(False)

    ax.set_facecolor("#e5ecf6")
    ax.tick_params(axis="x", rotation=45)
    # axs[1].set_facecolor('pink')
    # axs[2].set_facecolor('pink')
    #fig.tight_layout()
    ax.set_xlabel("Time")
    ax.set_ylabel("Animals")
    # ax.tick_params(axis="both", which="major", labelsize=6)
    # ax.tick_params(axis="both", which="minor", labelsize=6)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_")
    print("saving figure ", file_path)
    fig.savefig(file_path, bbox_inches="tight")
    # print("saved ", filename)
    # fig.savefig(file_path.replace(".png", ".svg"))
    # plt.show()

    # time_axis = np.array([pd.Timestamp(x).to_pydatetime() for x in time_axis])
    # print(time_axis)
    # # animal_ids_formatted_ent = ["\""+x+"\"" for x in animal_ids_formatted_ent]
    # #fig = make_subplots(rows=1, cols=1)
    # #cbarlocs = [0.89, 0.5, 0.11]
    #
    # html_formatted = []
    # for item in animal_ids_formatted_ent:
    #     formatted = item
    #     split = formatted.split()
    #     if "1to2" in item.lower():
    #         formatted = f"{''.join(split[0][7:])} <b>{split[-1]}</b>"
    #         html_formatted.append(formatted)
    #         continue
    #     if "2to2" in item.lower():
    #         formatted = f"{''.join(split[0][7:])} <b><i>{split[-1]}</i></b>"
    #         html_formatted.append(formatted)
    #         continue
    #     if "2to1" in item.lower():
    #         formatted = f"{''.join(split[0][7:])} <i>{split[-1]}</i>"
    #         html_formatted.append(formatted)
    #         continue
    #     if "3to2" in item.lower():
    #         formatted = f"{''.join(split[0][7:])} <i>{split[-1]}</i>"
    #         html_formatted.append(formatted)
    #         continue
    #
    #     formatted = f"{''.join(split[0][7:])}  {split[-1]}"
    #     html_formatted.append(formatted)
    #
    # a[0][0] = 0#workaround to prevent empty heatmap
    # fig_im_a_log_anscomb = go.Figure(
    #     data=go.Heatmap(
    #         z=a,
    #         x=time_axis,
    #         y=html_formatted,
    #         #colorbar=dict(len=0.25, y=cbarlocs[0]),
    #         colorscale="Viridis"
    #     )
    # )
    #
    # fig_im_a_log_anscomb.update_traces(showscale=False)
    # fig_im_a_log_anscomb.update_yaxes(type="category")
    # # fig_im_a_log_anscomb.update_traces(showscale=True)
    #
    # # fig.add_trace(fig_im_a_log_anscomb["data"][0], row=1, col=1)
    # # fig.update_layout(
    # #     #title=title1.replace("10T", "10 minutes"),
    # #     autosize=False,
    # #     width=900,
    # #     height=450,
    # # )
    #
    # # fig_im_a_log_anscomb.update_layout(
    # #     autosize=False,
    # #     width=900,
    # #     height=450,
    # # )
    #
    # # fig_im_a_i = go.Figure(data=go.Heatmap(
    # #     z=a_i,
    # #     x=time_axis,
    # #     y=html_formatted,
    # #     colorbar=dict(len=0.25, y=cbarlocs[1]),
    # #     colorscale='Viridis'))
    # # fig_im_a_i.update_yaxes(type='category')
    # # fig_im_a_i.update_traces(showscale=False)
    #
    # # fig.add_trace(fig_im_a_i['data'][0], row=2, col=1)
    # #
    # # fig_a_i_only = go.Figure(data=go.Heatmap(
    # #     z=a_i_only,
    # #     x=time_axis,
    # #     y=html_formatted,
    # #     colorbar=dict(len=0.25, y=cbarlocs[2]),
    # #     colorscale='Viridis'))
    # # fig_a_i_only.update_yaxes(type='category')
    # # # fig_a_i_only.update_traces(showscale=True)
    # #
    # # fig.add_trace(fig_a_i_only['data'][0], row=3, col=1)
    #
    # has_famdata = False
    # for item in animal_ids_formatted_ent:
    #     if "to" in item.lower():
    #         has_famdata = True
    #         break
    # filename = f"{range_id}_dataset_heatmap_{farm_id}_{resolution}_has_famacha_{has_famdata}.html"
    # out_DIR.mkdir(parents=True, exist_ok=True)
    # file_path = out_DIR / filename.replace("=", "_").lower()
    # print(file_path)
    # fig_im_a_log_anscomb.write_html(str(file_path))
    # return fig_im_a_log_anscomb


# def create_heatmap(
#     DATA,
#     k,
#     idx,
#     itot,
#     famacha_data,
#     day_before_famacha_test,
#     farm_id,
#     DATASET_INFO,
#     out_DIR,
#     n_job,
# ):
#     print("create_heatmap")
#     print(f"progress create_heatmap {idx}/{itot} ...")
#     # activity_list = []
#     time_axis = None
#     # animal_ids = []
#     entropy_list = []
#     entropy_s2_list = []
#     raw = []
#     raw_i = []
#     raw_e = []
#     raw_m = []
#     raw_bat = []
#     raw_ss = []
#     raw_xmin = []
#     raw_xmax = []
#     raw_ymin = []
#     raw_ymax = []
#     raw_zmin = []
#     raw_zmax = []
#     resolution = ""
#     wid = 0
#     range_id = ""
#     for item in DATA:
#         # animal_ids.append(item[0])
#         time_axis = item[k][1]
#         entropy_list.append(item[k][2])
#         entropy_s2_list.append(item[k][3])
#         raw.append(item[k][4])
#         raw_i.append(item[k][5])
#         raw_e.append(item[k][6])
#         raw_m.append(item[k][7])
#
#         raw_bat.append(item[k][8])
#         raw_ss.append(item[k][9])
#         raw_xmin.append(item[k][10])
#         raw_xmax.append(item[k][11])
#         raw_ymin.append(item[k][12])
#         raw_ymax.append(item[k][13])
#         raw_zmin.append(item[k][14])
#         raw_zmax.append(item[k][15])
#
#         resolution = item[k][16]
#         wid = item[k][17]
#         range_id = item[k][18]
#
#     # if range_id == "161280_171360" or idx == 16:
#     #     time_axis = np.array([pd.Timestamp(x).to_pydatetime() for x in time_axis])
#     #     print(time_axis)
#     #     print(0)
#
#     df_raw = pd.DataFrame(raw, dtype=object)
#     df_raw[0][0] = 0
#     print(df_raw)
#     c = np.nansum(df_raw.iloc[:, :-4].values.astype(float))
#     # print("c", c)
#     # if c == 0:
#     #     print("empty df only Nan!")
#     #     df_raw[0][0] = 0 #need at leas 1 value in dataframe for heatmap plot otherwise xaxis ignored
#         #return
#     df_raw_i = pd.DataFrame(raw_i, dtype=object)
#     # df_raw_e = pd.DataFrame(raw_e, dtype=object)
#     # df_raw_m = pd.DataFrame(raw_m, dtype=object)
#
#     # df_raw_bat = pd.DataFrame(raw_bat, dtype=object)
#     # df_raw_ss = pd.DataFrame(raw_ss, dtype=object)
#     #
#     # df_raw_xmin = pd.DataFrame(raw_xmin, dtype=object)
#     # df_raw_xmax = pd.DataFrame(raw_xmax, dtype=object)
#     #
#     # df_raw_ymin = pd.DataFrame(raw_ymin, dtype=object)
#     # df_raw_ymax = pd.DataFrame(raw_ymax, dtype=object)
#     #
#     # df_raw_zmin = pd.DataFrame(raw_zmin, dtype=object)
#     # df_raw_zmax = pd.DataFrame(raw_zmax, dtype=object)
#
#     header = [x for x in range(df_raw.shape[1])]
#     header[-1] = "id"
#     header[-2] = "entropy_s2"
#     header[-3] = "entropy"
#
#     print(f"add_famacha_format_id_todf... {idx}/{itot} ...")
#     df_raw = add_famacha_format_id_todf(df_raw, header, famacha_data)
#     df_raw_i = add_famacha_format_id_todf(df_raw_i, header, famacha_data)
#     # df_raw_ss = add_famacha_format_id_todf(df_raw_ss, header, famacha_data)
#     # df_raw_bat = add_famacha_format_id_todf(df_raw_bat, header, famacha_data)
#     # df_raw_xmin = add_famacha_format_id_todf(df_raw_xmin, header, famacha_data, s2=True)
#     # df_raw_xmax = add_famacha_format_id_todf(df_raw_xmax, header, famacha_data, s2=True)
#     # df_raw_ymin = add_famacha_format_id_todf(df_raw_ymin, header, famacha_data, s2=True)
#     # df_raw_ymax = add_famacha_format_id_todf(df_raw_ymax, header, famacha_data, s2=True)
#     # df_raw_zmin = add_famacha_format_id_todf(df_raw_zmin, header, famacha_data, s2=True)
#     # df_raw_zmax = add_famacha_format_id_todf(df_raw_zmax, header, famacha_data, s2=True)
#     print(f"add_famacha_format_id_todf done {idx}/{itot} ...")
#
#     if c != 0:
#         df_raw = df_raw[df_raw["possible"] == False]
#         df_raw_i = df_raw_i[df_raw_i["possible"] == False]
#
#     # if df_raw.shape[0] == 0:
#     #     print("empty df!")
#     #     return
#
#     # df_raw_e = df_raw_e[df_raw_e["possible"] == False]
#     # df_raw_m = df_raw_m[df_raw_m["possible"] == False]
#     # df_raw_ss = df_raw_ss[df_raw_ss["possible"] == False]
#     # df_raw_bat = df_raw_bat[df_raw_bat["possible"] == False]
#     # df_raw_xmin = df_raw_xmin[df_raw_xmin["possible"] == False]
#     # df_raw_xmax = df_raw_xmax[df_raw_xmax["possible"] == False]
#     # df_raw_ymin = df_raw_ymin[df_raw_ymin["possible"] == False]
#     # df_raw_ymax = df_raw_ymax[df_raw_ymax["possible"] == False]
#     # df_raw_zmin = df_raw_zmin[df_raw_zmin["possible"] == False]
#     # df_raw_zmax = df_raw_zmax[df_raw_zmax["possible"] == False]
#     print(f"ready for figure {idx}/{itot} ...")
#
#     # n = 3
#     # h = (df_raw.shape[0] * 40 * n) / 100
#     # w = 36.20 * 2
#     # # fig, axs = plt.subplots(n, figsize=(w, h))
#     # # for p in range(n):
#     # #     axs[p].yaxis.set_label_position("right")
#     # #     axs[p].yaxis.tick_right()
#     #
#
#     annotation, missing_ids = create_annotation_matrix(
#         df_raw, time_axis, day_before_famacha_test
#     )
#     print("ANNOT", np.unique(annotation.flatten()).tolist())
#     # f_id = [list(x)[1] for x in list(map(set, annotation))]
#     f_id = []
#     for x in list(map(set, annotation)):
#         if len(x) != 2:
#             f_id.append("NaN")
#             continue
#         f_id.append(list(x)[1])
#
#     a = df_raw.iloc[:, :-5].values
#     a_i = df_raw_i.iloc[:, :-5].values
#     a_i_only = a_i.copy()
#     a_i_only[np.isnan(a) == False] = np.nan
#     # ss = df_raw_ss.iloc[:, :-5].values
#     # bat = df_raw_bat.iloc[:, :-5].values
#     #
#     #
#     # a_xmin = df_raw_xmin.iloc[:, :-5].values
#     # a_xmax = df_raw_xmax.iloc[:, :-5].values
#     # a_ymin = df_raw_ymin.iloc[:, :-5].values
#     # a_ymax = df_raw_ymax.iloc[:, :-5].values
#     # a_zmin = df_raw_zmin.iloc[:, :-5].values
#     # a_zmax = df_raw_zmax.iloc[:, :-5].values
#
#     # viridis = cm.get_cmap('viridis', 256)
#     # newcolors = viridis(np.linspace(0, 1, 256))
#     # pink = np.array([0 / 256, 0 / 256, 0 / 256, 1])
#     # newcolors[:1, :] = pink
#     # newcmp = ListedColormap(newcolors)
#
#     # im_a_log_anscomb = axs[0].imshow(a, cmap=newcmp, aspect='auto',
#     #                                  interpolation="nearest",
#     #                                  extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
#     # plt.colorbar(im_a_log_anscomb, ax=axs[0])
#     # axs[0].xaxis_date()
#
#     # im_a_i = axs[1].imshow(a_i, cmap=newcmp, aspect='auto',
#     #                                  interpolation="nearest",
#     #                                  extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
#     # plt.colorbar(im_a_i, ax=axs[1])
#     # axs[1].xaxis_date()
#
#     # im_a_i_only = axs[2].imshow(a_i_only, cmap=newcmp, aspect='auto',
#     #                                  interpolation="nearest",
#     #                                  extent=[x_lims[0], x_lims[-1], 0, df_raw.iloc[:, :-4].values.shape[0]])
#     # plt.colorbar(im_a_i_only, ax=axs[2])
#     # axs[2].xaxis_date()
#
#     # if resolution == "1T":
#     #     for p in range(n):
#     #         axs[p].xaxis_date()
#     #         axs[p].xaxis.set_major_formatter(date_format)
#     #         axs[p].xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
#     #
#     # elif resolution == "1D":
#     #     for p in range(n):
#     #         # axs[p].xaxis_date()
#     #         axs[p].xaxis.set_major_formatter(date_format)
#     #         axs[p].xaxis.set_major_locator(mdates.DayLocator(interval=7))
#     # else:
#     #     for p in range(n):
#     #         # axs[p].xaxis_date()
#     #         axs[p].xaxis.set_major_formatter(date_format)
#     #         axs[p].xaxis.set_major_locator(mdates.DayLocator(interval=7))
#     #
#     # fig.autofmt_xdate()
#
#     animal_ids_formatted_ent = df_raw["id"].values[::-1]
#     animal_ids_formatted_ent = np.array(
#         [x[1] + " " + x[0] for x in zip(f_id, animal_ids_formatted_ent)]
#     )
#     # animal_ids_formatted_ent_s2 = df_raw_xmin["id"].values[::-1]
#     # for p in range(n):
#     #     axs[p].set_yticklabels(animal_ids_formatted_ent)
#     #     axs[p].set_yticks(np.arange(len(animal_ids_formatted_ent)))
#     #     if p > 2:
#     #         axs[p].set_yticklabels(animal_ids_formatted_ent_s2)
#     #         axs[p].set_yticks(np.arange(len(animal_ids_formatted_ent_s2)))
#     #
#     #
#     # for i in range(len(axs[0].get_yticklabels())):
#     #     if "*" in str(animal_ids_formatted_ent[i]):
#     #         for p in range(n):
#     #             axs[p].get_yticklabels()[i].set_color("tab:red")
#
#     # print("adding annotations...")
#     # activity_list_matrix = df_raw.iloc[:, :-4].values
#     # for i in range(activity_list_matrix.shape[0]):
#     #     cpt = 0
#     #     for j in range(activity_list_matrix.shape[1]):
#     #         an = annotation[i, j]
#     #         if an == "n" or an == "":
#     #             cpt = 0
#     #             continue
#     #         cpt += 1
#     #         color = "lightgrey"
#     #
#     #         if an == "1To1":
#     #             color = "white"
#     #         if an == "1To2":
#     #             color = "red"
#     #         if an == "2To2":
#     #             color = "blue"
#     #         if an == "2To1":
#     #             color = "orange"
#     #         if an == "3To2":
#     #             color = "lawngreen"
#     #         #use ASCII 219 for text highlight instead of rectangle
#     #         offset_y = 0.6
#     #         offset_x = 0.9
#     #         # ax.text(j-offset_x, activity_list_matrix.shape[0] - i - offset_y, "x", ha="left", va="baseline", color=color, alpha=0.4, fontsize=8, fontweight='bold')
#     #         w = day_before_famacha_test
#     #         lw = 1.4
#     #         if cpt == 1:
#     #             for p in range(n):
#     #                 rec = Rectangle((x_lims[j], activity_list_matrix.shape[0] - i - 1), w, 0.85, fill=False,
#     #                                 edgecolor=color, facecolor=None, lw=lw, alpha=1)
#     #                 axs[p].add_patch(rec)
#
#     param_str = (
#         f"sampling={resolution} day_before_famacha_test={day_before_famacha_test}"
#     )
#     ntrans_with_samples = len(animal_ids_formatted_ent) - len(missing_ids)
#
#     title1 = f"Activity data per {resolution} {farm_id} herd"
#
#     # title2 = f"Imputed Activity data per {resolution}  {farm_id} herd and dataset samples location\n{breaklineinsert(str(DATASET_INFO))}\n{param_str}\n*no famacha data corresponding animal id size={len(missing_ids)}/{len(animal_ids_formatted_ent)}\ntransponder traces with fam samples={ntrans_with_samples}"
#     #
#     # title3 = f"(imputed only) Imputed Activity raw data per {resolution}  {farm_id} herd and dataset samples location\n{breaklineinsert(str(DATASET_INFO))}\n{param_str}\n*no famacha data corresponding animal id size={len(missing_ids)}/{len(animal_ids_formatted_ent)}\ntransponder traces with fam samples={ntrans_with_samples}"
#
#     # patch1 = mpatches.Patch(color='white', label="1To1 "+str(DATASET_INFO["1To1"]))
#     # patch2 = mpatches.Patch(color='red', label="1To2 "+str(DATASET_INFO["1To2"]))
#     # patch3 = mpatches.Patch(color='blue', label="2To2 "+str(DATASET_INFO["2To2"]))
#     # patch4 = mpatches.Patch(color='orange', label="2To1 "+str(DATASET_INFO["2To1"]))
#     # patch5 = mpatches.Patch(color='lawngreen', label="3To2 "+str(DATASET_INFO["3To2"]))
#
#     # for p in range(n):
#     #     axs[p].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.5)
#     #     # axs[1].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.5)
#     #     # axs[2].legend(handles=[patch1, patch2, patch3, patch4, patch5], loc='lower left', fancybox=True, framealpha=0.6)
#     #
#     #     axs[p].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
#     #     # axs[1].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
#     #     # axs[2].yaxis.set(ticks=np.arange(0.5, len(animal_ids_formatted_ent)))
#     #
#     #     axs[p].set_facecolor('pink')
#     #     # axs[1].set_facecolor('pink')
#     #     # axs[2].set_facecolor('pink')
#
#     # fig.tight_layout()
#
#     # file_path = out_DIR +"/"+ filename.replace("=", "_")
#     # print("saving figure ", file_path)
#     # fig.savefig(file_path, bbox_inches='tight')
#     # print("saved ", filename)
#     # fig.savefig(file_path.replace(".png", ".svg"))
#     # plt.interactive(True)
#     # plt.show()
#     # date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
#     # x_lims = mdates.date2num(time_axis)
#
#     time_axis = np.array([pd.Timestamp(x).to_pydatetime() for x in time_axis])
#     print(time_axis)
#     # animal_ids_formatted_ent = ["\""+x+"\"" for x in animal_ids_formatted_ent]
#     #fig = make_subplots(rows=1, cols=1)
#     #cbarlocs = [0.89, 0.5, 0.11]
#
#     html_formatted = []
#     for item in animal_ids_formatted_ent:
#         formatted = item
#         split = formatted.split()
#         if "1to2" in item.lower():
#             formatted = f"{''.join(split[0][7:])} <b>{split[-1]}</b>"
#             html_formatted.append(formatted)
#             continue
#         if "2to2" in item.lower():
#             formatted = f"{''.join(split[0][7:])} <b><i>{split[-1]}</i></b>"
#             html_formatted.append(formatted)
#             continue
#         if "2to1" in item.lower():
#             formatted = f"{''.join(split[0][7:])} <i>{split[-1]}</i>"
#             html_formatted.append(formatted)
#             continue
#         if "3to2" in item.lower():
#             formatted = f"{''.join(split[0][7:])} <i>{split[-1]}</i>"
#             html_formatted.append(formatted)
#             continue
#
#         formatted = f"{''.join(split[0][7:])}  {split[-1]}"
#         html_formatted.append(formatted)
#
#     a[0][0] = 0#workaround to prevent empty heatmap
#     fig_im_a_log_anscomb = go.Figure(
#         data=go.Heatmap(
#             z=a,
#             x=time_axis,
#             y=html_formatted,
#             #colorbar=dict(len=0.25, y=cbarlocs[0]),
#             colorscale="Viridis"
#         )
#     )
#
#     fig_im_a_log_anscomb.update_traces(showscale=False)
#     fig_im_a_log_anscomb.update_yaxes(type="category")
#     # fig_im_a_log_anscomb.update_traces(showscale=True)
#
#     # fig.add_trace(fig_im_a_log_anscomb["data"][0], row=1, col=1)
#     # fig.update_layout(
#     #     #title=title1.replace("10T", "10 minutes"),
#     #     autosize=False,
#     #     width=900,
#     #     height=450,
#     # )
#
#     # fig_im_a_log_anscomb.update_layout(
#     #     autosize=False,
#     #     width=900,
#     #     height=450,
#     # )
#
#     # fig_im_a_i = go.Figure(data=go.Heatmap(
#     #     z=a_i,
#     #     x=time_axis,
#     #     y=html_formatted,
#     #     colorbar=dict(len=0.25, y=cbarlocs[1]),
#     #     colorscale='Viridis'))
#     # fig_im_a_i.update_yaxes(type='category')
#     # fig_im_a_i.update_traces(showscale=False)
#
#     # fig.add_trace(fig_im_a_i['data'][0], row=2, col=1)
#     #
#     # fig_a_i_only = go.Figure(data=go.Heatmap(
#     #     z=a_i_only,
#     #     x=time_axis,
#     #     y=html_formatted,
#     #     colorbar=dict(len=0.25, y=cbarlocs[2]),
#     #     colorscale='Viridis'))
#     # fig_a_i_only.update_yaxes(type='category')
#     # # fig_a_i_only.update_traces(showscale=True)
#     #
#     # fig.add_trace(fig_a_i_only['data'][0], row=3, col=1)
#
#     has_famdata = False
#     for item in animal_ids_formatted_ent:
#         if "to" in item.lower():
#             has_famdata = True
#             break
#     filename = f"{range_id}_dataset_heatmap_{farm_id}_{resolution}_has_famacha_{has_famdata}.html"
#     out_DIR.mkdir(parents=True, exist_ok=True)
#     file_path = out_DIR / filename.replace("=", "_").lower()
#     print(file_path)
#     fig_im_a_log_anscomb.write_html(str(file_path))
#     return fig_im_a_log_anscomb


def main(
    output: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    activity_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    farm_id: str = "farm",
    sampling: str = "T",
    activity_col: str = "first_sensor_value",
    day_before_famacha_test: int = 7,
    w: int = -1,
    res: str = "60T",
    start: int = 0,
    end: str = -1,
    no_filter: bool = False,
    display_famacha: bool = True,
    zoom: bool = False,
    weight: int = 0,
    height: int = 0,
    n_job: int = 6,
):
    """Create heatmap of the heard with sample overlay.\n
    Args:\n
        output: Directory path of heatmap output. (Directory will be created if does not exist)
        activity_dir: Directory with activity data
        dataset_dir: Path of the directory containing dataset .csv and class info .txt.
        w: Size of slicing window in minutes. (pass negative value for entire signal trace)
        res: Sampling resolution.
        start: start time in minute.
        end: end time in minute.
        n_job: Number of thread to use.
    """

    try:
        dataset_file_path = glob.glob(str(dataset_dir / "*.csv"))[0]
    except IndexError as e:
        print(e)
        print(f"could not find dataset (.csv) file in {dataset_dir}")
    info_files = glob.glob(str(dataset_dir / "*.json"))
    info_file = None
    for file in info_files:
        info_file = file
        print("found info file %s" % file)
        break
    if info_file is None:
        raise IOError(f"missing dataset info (.json) file! in {dataset_dir}")
    DATASET_INFO = json.load(open(info_file))

    # farm_id, sampling, day_before_famacha_test = parse_options(dataset_file_path)

    out_DIR = output / farm_id
    out_DIR.mkdir(parents=True, exist_ok=True)

    print(f"farm_id{farm_id}")
    print(f"sampling{sampling}")
    print(f"day_before_famacha_test{day_before_famacha_test}")

    files = [str(x) for x in list(activity_dir.glob("*.csv")) if len(x.stem) > 5]
    if len(files) == 0:
        raise IOError(f"missing activity files .csv! in {activity_dir}")

    # get FAMACHA data
    famacha_data = {}
    dataset = load_dataset(dataset_file_path)
    for i in range(dataset.shape[0]):
        row = dataset.iloc[i, :]
        date_of_famacha_test_str = row["dtf1"].strip("'")
        date_of_famacha_test = pd.to_datetime(
            date_of_famacha_test_str, format="%d/%m/%Y"
        )
        label = row["label"]
        animal_id = row["serial"]
        if animal_id in famacha_data.keys():
            famacha_data[animal_id].append([date_of_famacha_test, label])
        else:
            famacha_data[animal_id] = [[date_of_famacha_test, label]]
    ######################################################
    pool = Pool(processes=n_job)
    results = []
    for i, file in enumerate(files):
        if "median" in file or "mean" in file:
            continue
        results.append(
            pool.apply_async(
                process_activity_data,
                (activity_col, file, i, len(files), w, res, start, end, zoom),
            )
        )
    pool.close()
    pool.join()
    pool.terminate()

    DATA = []
    for res in results:
        DATA.append(res.get())

    create_heatmap(
        display_famacha,
        activity_col,
        zoom,
        no_filter,
        f"heatmap_{farm_id}_{activity_col}_{zoom}.png",
        DATA,
        0,
        0,
        len(DATA[0]),
        famacha_data,
        day_before_famacha_test,
        farm_id,
        DATASET_INFO,
        out_DIR,
        n_job,
        weight,
        height
    )

    ######################################################
    # print("starting second pool.")
    # # MAX_THREADC = 6
    # MAX_THREADC = n_job
    # njob = MAX_THREADC if n_job >= MAX_THREADC else n_job
    # # pool2 = Pool(processes=njob)
    # print("with njob=%d" % njob)
    # r_ = list(range(len(DATA[0])))
    #
    # traces = []
    # for i, k in enumerate(r_):
    #     #print("feeding pool", i)
    #     # print(k, i, len(DATA[0]), day_before_famacha_test, farm_id, out_DIR)
    #     trace = create_heatmap(
    #         DATA,
    #         k,
    #         i,
    #         len(DATA[0]),
    #         famacha_data,
    #         day_before_famacha_test,
    #         farm_id,
    #         DATASET_INFO,
    #         out_DIR,
    #         n_job,
    #     )
    #     traces.append(trace)
    #
    #     # if i > 11:
    #     #     break
    # #     pool2.apply_async(create_heatmap, (DATA, k, i, len(DATA[0]), famacha_data, day_before_famacha_test, farm_id, DATASET_INFO, out_DIR, args.n_job))
    # # pool2.close()
    # # pool2.join()
    # # pool2.terminate()
    # figures_to_html(traces, out_DIR)

    print("done.")


def local_run():

    # main(
    #     output=Path("E:/thesis/heatmaps/raw_all_famacha_test"),
    #     activity_dir=Path(
    #         "F:/Data2/backfill_1min_xyz_delmas_fixed"
    #     ),
    #     dataset_dir=Path("E:/thesis/datasets/delmas/raw_all_famacha_test"),
    #     activity_col="signal_strength",
    #     farm_id="delmas",
    #     day_before_famacha_test=7,
    #     no_filter=False,
    #     display_famacha=False,
    #     res='1T',
    #     zoom=True
    # )

    main(
        output=Path("E:/thesis/heatmaps/raw_all_famacha_test"),
        activity_dir=Path(
            "F:/Data2/backfill_1min_xyz_delmas_fixed"
        ),
        dataset_dir=Path("E:/thesis/datasets/delmas/raw_all_famacha_test"),
        activity_col="signal_strength",
        farm_id="delmas",
        day_before_famacha_test=7,
        no_filter=False,
        display_famacha=False,
        res='1T',
        zoom=True,
        weight=9,
        height=2
    )

    main(
        output=Path("E:/thesis/heatmaps/raw_all_famacha_test"),
        activity_dir=Path(
            "F:/Data2/backfill_1min_xyz_delmas_fixed"
        ),
        dataset_dir=Path("E:/thesis/datasets/delmas/raw_all_famacha_test"),
        activity_col="first_sensor_value",
        farm_id="delmas",
        day_before_famacha_test=7,
        no_filter=False,
        display_famacha=False,
        res='1T',
        zoom=True,
        weight=9,
        height=2
    )

    # main(
    #     output=Path("E:/thesis/heatmaps/raw_all_famacha_test"),
    #     activity_dir=Path(
    #         "E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed_with_missing_tag"
    #     ),
    #     dataset_dir=Path("E:/thesis/datasets/cedara/raw_all_famacha_test"),
    #     activity_col="first_sensor_value",
    #     farm_id="cedara",
    #     day_before_famacha_test=7,
    #     no_filter=True,
    #     display_famacha=False,
    #     zoom=True
    # )

    # main(
    #     output=Path("E:/thesis/heatmaps/raw_all_famacha_test"),
    #     activity_dir=Path(
    #         "E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed_with_missing_tag"
    #     ),
    #     dataset_dir=Path("E:/thesis/datasets/delmas/raw_all_famacha_test"),
    #     activity_col="first_sensor_value",
    #     farm_id="delmas",
    #     day_before_famacha_test=7,
    #     no_filter=True,
    #     display_famacha=False
    # )




    # main(
    #     output=Path("E:/thesis/heatmaps/raw_usable"),
    #     activity_dir=Path("E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed"),
    #     dataset_dir=Path("E:/thesis/datasets/cedara/datasetraw_none_7day_clipped"),
    #     activity_col="first_sensor_value",
    #     farm_id="cedara",
    #     day_before_famacha_test=7,
    # )
    #
    # main(
    #     output=Path("E:/thesis/heatmaps/raw_usable"),
    #     activity_dir=Path("E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed"),
    #     dataset_dir=Path("E:/thesis/datasets/delmas/datasetraw_none_7day"),
    #     activity_col="first_sensor_value",
    #     farm_id="delmas",
    #     day_before_famacha_test=7,
    # )

    # main(output=Path("E:/thesis2/heatmap"),
    #      activity_dir=Path("F:/Data2/backfill_1min_delmas_fixed/delmas_70101200027"),
    #      dataset_dir=Path("E:/thesis_debug/dataset/cedara"),
    #      farm_id="delmas",
    #      activity_col="first_sensor_value",
    #      day_before_famacha_test=7)

    # main(
    #     output=Path("E:/thesis2/heatmap"),
    #     activity_dir=Path(
    #         "F:/MRNN/imputed_data/4_missingrate_[0.0]_seql_1440_iteration_100_hw__n_421"
    #     ),
    #     dataset_dir=Path("E:/Data2/debug3/delmas/datasetraw_none_7day"),
    #     activity_col="first_sensor_value_mrnn",
    #     farm_id="delmas_imputed",
    #     day_before_famacha_test=7,
    # )

    # main(output=Path("E:/thesis2/heatmap"),
    #      activity_dir=Path("F:/Data2/backfill_1min_cedara_fixed"),
    #      dataset_dir=Path("E:/Data2/debug3/cedara/dataset6_mrnn_7day"),
    #      farm_id="cedara_imputed",
    #      day_before_famacha_test=7)


if __name__ == "__main__":
    local_run()
    # typer.run(main)
