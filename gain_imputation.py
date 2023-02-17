"""Main function for UCI letter and spam datasets.
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import numpy as np

np.random.seed(0)  # for reproducability

from gain.gain import gain
from gain.helper import (
    binary_sampler,
    linear_interpolation_v,
    reshape_matrix_v1,
    reshape_matrix_v2,
    build_formated_axis,
)
from utils.Utils import create_rec_dir, inverse_anscombe

# matplotlib.use('Qt5Agg')
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 100000000000
import datetime as dt
import scipy.stats
from utils.Utils import anscombe
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from var import *


def entropy_(to_resample):
    e = 0
    if to_resample.dropna().size > 0:
        e = scipy.stats.entropy(to_resample.dropna())
    return e


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def read_activity_data(file, i, nfiles, window):
    print("reading files %d/%d  ..." % (i, nfiles))
    animal_id = file.stem
    df_activity = pd.read_csv(file, sep=",")
    if window:
        print("WINDOW ON !!!")
        w = 1440 * 7 * 12
        start = 413129
        end = start + w
        df_activity = df_activity.loc[start:end, :]
    return animal_id, df_activity


def plot_imputed_data(
    out, imputed_data_x_gain, imputed_data_x_li, raw_data, ori_data_x, ids, timestamps
):
    print("plot_imputed_data...")
    out = out + "/figures/imp/"
    create_rec_dir(out)
    w = 1440 * 3
    start = 413129
    end = start + w
    time_axis = np.array([dt.datetime.fromtimestamp(ts) for ts in timestamps])[
        start:end
    ]

    for i in range(imputed_data_x_li.shape[1]):

        imputed_li = imputed_data_x_li[:, i][start:end]
        imputed_gain = imputed_data_x_gain[:, i][start:end]
        original = ori_data_x[:, i][start:end]
        # raw = raw_data[:, i]

        nan_count = np.count_nonzero(np.isnan(original))
        if (
            nan_count == original.size
            or all(x == imputed_li[0] for x in imputed_li)
            or all(x == original[0] for x in original)
        ):
            continue

        id = ids[i]
        date_format = mdates.DateFormatter("%d/%b/%Y %H:%M")
        # plt.clf()
        # plt.cla()
        # fig, ax = plt.subplots(3)
        # ax[1].xaxis_date()
        # ax[1].xaxis.set_major_formatter(date_format)
        # ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=60))
        # ax[1].tick_params(axis='x', rotation=25)
        # w = 500
        # # ax[1].bar(time_axis[0:w], imputed_li[0:w], label="after li gain", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], original[0:w], label="original", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], imputed_gain[0:w], label="after gain gain", alpha=0.5, width=0.1)
        #
        # ax[1].plot(list(range(w)), imputed_li[0:w], label="after li gain", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), original[0:w], label="original", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), imputed_gain[0:w], label="after gain gain", alpha=0.5, marker='o')
        #
        # ax[1].set_title('Transformed activity before and after gain')
        # ax[1].legend()
        #
        # ax[2].plot(time_axis[0:w], np.abs(original[0:w] - imputed_gain[0:w]), label="original - gain", alpha=0.5, color='blue', linestyle='-')
        # ax[2].plot(time_axis[0:w], np.abs(original[0:w] - imputed_li[0:w]), label="original - linear interp", alpha=0.5, color='red', linestyle='-')
        # ax[2].legend()
        #
        # ax[0].plot(time_axis[0:w], raw[0:w], label="raw")
        # # ax[0].set_title('Raw')

        rmse_gain = int(np.nansum(np.abs(original - imputed_gain)))
        rmse_li = int(np.nansum(np.abs(original - imputed_li)))
        # fig.suptitle('gain %d      li %d' % (rmse_gain, rmse_li), fontsize=14)

        filename = "%d_gain_%d_li_%d.png" % (id, rmse_gain, rmse_li)
        filepath = "%s/%s" % (out, filename)
        # print('saving fig...')

        df = pd.DataFrame()
        df["time"] = time_axis.tolist() + time_axis.tolist() + time_axis.tolist()
        df["data"] = original.tolist() + imputed_li.tolist() + imputed_gain.tolist()
        df["gain"] = (
            ["ORIGINAL" for _ in range(len(original))]
            + ["LI" for _ in range(len(original))]
            + ["GAIN" for _ in range(len(original))]
        )
        color = []
        for ii in range(time_axis.size):
            o = original[ii]
            ili = imputed_li[ii]
            igain = imputed_gain[ii]
            if o == ili == igain:
                color.append("R")
                continue
            if np.isnan(o):
                color.append("")
                continue
            color.append("A_M")

        df["color"] = color + color + color

        # fig_px = px.bar(df, x="time", y="data", color='gain', height=900, text='color', barmode="group", title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li))
        fig_px = px.line(
            df,
            x="time",
            y="data",
            color="gain",
            height=900,
            text="color",
            title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li),
        )
        # fig_px.update_traces(textposition='inside', insidetextanchor="start")
        # fig_px.add_scatter(x=time_axis.tolist(), y=np.abs(original - imputed_li).tolist(), name="abs(original - imputed_li)", mode='lines+markers', marker=dict(color='coral'), connectgaps=True)
        # fig_px.add_scatter(x=time_axis.tolist(), y=np.abs(original - imputed_gain).tolist(), name="abs(original - imputed_gain)", mode='lines+markers', marker=dict(color='green'), connectgaps=True)

        # fig.tight_layout()
        # fig.savefig(filepath)
        fig_px.write_html(filepath.replace(".png", ".html"))

        # plt.close(fig)
        # fig.show()
        # print("saved!")

        # plt.clf()
        # plt.cla()
        # fig, ax = plt.subplots(3)
        # b = int(len(raw)/10)
        # ax[0].hist(raw, bins=b, density=False)
        # ax[0].set_title("Histogram of raw data " + str(id))
        # ax[1].hist(original, bins=b, density=False)
        # ax[1].set_title("Histogram of anscombe of raw data " + str(id))
        # ax[2].hist(imputed_li, bins=b, density=False)
        # ax[2].set_title("Histogram of imputed data " + str(id))
        # fig.tight_layout()
        # filename = "hist_%d.png" % id
        # filepath = "%s/%s" % (out, filename)
        # # print('saving fig...')
        # try:
        #     plt.savefig(filepath)
        # except OverflowError as e:
        #     print(e)
        # plt.close(fig)


def worker_export_heatmaps(
    i,
    tot,
    transponder,
    n_top_traces,
    N_TRANSPOND,
    ss,
    ss_t,
    first_timestamp,
    THRESH_DT,
    out,
    miss_rate,
    export_heatmaps
):
    d_t = transponder.iloc[:, : -n_top_traces - 2]
    xaxix_label, yaxis_label = build_formated_axis(
        first_timestamp, min_in_row=d_t.shape[1], days_in_col=d_t.shape[0]
    )
    valid = np.sum((~np.isnan(d_t.values)).astype(int))
    if valid <= 0:
        return
    # ss = ss_reshaped[:, :-N_TRANSPOND - 1]
    # ss_t = dfs_ss[i].iloc[:, :-n_top_traces - 2]

    id = int(transponder["id"].values[0])

    if miss_rate == 0:
        if export_heatmaps:
            print(f"exporting heatmap {id} {i}/{tot}...")

    xaxix_label[0] = xaxix_label[0].replace(minute=0)
    fig = go.Figure(
        data=go.Heatmap(z=d_t, x=xaxix_label, y=yaxis_label, colorscale="Viridis")
    )
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_yaxes(tickformat="%d %b %Y")
    fig.update_layout(
        title="Transponder id %d thresh=%d" % (id, THRESH_DT),
        yaxis_title="Date(day)",
        xaxis_title="Time (1 min bins)",
    )
    filename = out / f"{id}_reshaped_{THRESH_DT}_{valid}_filtered.html"
    # if i % 100 == 0:
    if miss_rate == 0:
        if export_heatmaps:
            print(filename)
            fig.write_html(str(filename))

    d_t_na_li = d_t.interpolate(method="linear", limit_direction="both", axis=1)
    fig = go.Figure(
        data=go.Heatmap(
            z=d_t_na_li,
            x=xaxix_label,
            y=np.arange(0, d_t_na_li.shape[1]),
            colorscale="Viridis",
        )
    )
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_yaxes(tickformat="%d %b %Y")
    fig.update_layout(
        title="Transponder id %d thresh=%d" % (id, THRESH_DT),
        yaxis_title="Samples",
        xaxis_title="Time (1 min bins)",
    )
    filename = out / f"{id}_reshaped_{THRESH_DT}_{valid}_li.html"
    # if i % 100 == 0:
    if miss_rate == 0:
        if export_heatmaps:
            print(filename)
            fig.write_html(str(filename))
    return xaxix_label, yaxis_label


def worker_export_csv(i, id, tot, timestamp, date_str, ori_data_x, idata, ildata, out):
    print("progress %d/%d ..." % (i, tot))
    # id = str(int(ids[i]))
    df = pd.DataFrame()
    df["timestamp"] = timestamp
    df["date_str"] = date_str
    df["first_sensor_value"] = np.array(
        [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(ori_data_x), 0)]
    )
    df["first_sensor_value_gain"] = np.array(
        [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(idata), 0)]
    )[0 : df["first_sensor_value"].shape[0]]
    df["first_sensor_value_li"] = np.array(
        [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(ildata), 0)]
    )[0 : df["first_sensor_value"].shape[0]]
    df["imputed"] = np.isnan(df["first_sensor_value"]).astype(int)
    filename = id + ".csv"
    filepath = out / filename
    df.to_csv(filepath, sep=",", index=False)
    print(filepath)


def export_imputed_data(
    out,
    data_m_x,
    ori_data_x,
    idata,
    ildata,
    timestamp,
    date_str,
    ids,
    alpha,
    hint,
    n_job,
):
    print("exporting imputed data...")
    print(ids)
    data_m_x[np.isnan(data_m_x)] = 1

    pool = Pool(processes=n_job)
    for i in range(idata.shape[1]):
        id = str(int(ids[i]))
        pool.apply_async(
            worker_export_csv,
            (
                i,
                id,
                len(ids),
                timestamp.values,
                date_str.values,
                ori_data_x[:, i],
                idata[:, i],
                ildata[:, i],
                out,
            ),
        )
    pool.close()
    pool.join()
    pool.terminate()

    # for i in range(idata.shape[1]):
    #     print("progress %d/%d ..." % (i, len(ids)))
    #     id = str(int(ids[i]))
    #     df = pd.DataFrame()
    #     df["timestamp"] = timestamp.values
    #     df["date_str"] = date_str.values
    #     df["first_sensor_value"] = np.array(
    #         [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(ori_data_x[:, i]), 0)])
    #     df["first_sensor_value_gain"] = np.array(
    #         [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(idata[:, i]), 0)])[0:df["first_sensor_value"].shape[0]]
    #     df["first_sensor_value_li"] = np.array(
    #         [x if np.isnan(x) else int(x) for x in inverse_anscombe(np.exp(ildata[:, i]), 0)])[0:df["first_sensor_value"].shape[0]]
    #     df["imputed"] = (idata[:, i] >= 0).astype(int)[0:df["first_sensor_value"].shape[0]]
    #     filename = id + ".csv"
    #     filepath = out / filename
    #     df.to_csv(filepath, sep=',', index=False)
    #     print(filepath)


def load_farm_data(
    fname,
    n_job,
    n_top_traces=-1,
    enable_anscombe=False,
    enable_log_anscombe=False,
    enable_remove_zeros=False,
    window=False,
):
    print("load_farm_data...")
    data_folder = Path(fname)
    if "delmas" in fname.lower():
        if n_top_traces <= 0:
            files = [data_folder / f"{x}.csv" for x in transponders_delmas]
        else:
            files = [
                data_folder / f"{x}.csv" for x in transponders_delmas[0:n_top_traces]
            ]

    if "cedara" in fname.lower():
        if n_top_traces <= 0:
            files = [data_folder / f"{x}.csv" for x in transponders_cedara]
        else:
            files = [
                data_folder / f"{x}.csv" for x in transponders_cedara[0:n_top_traces]
            ]

    # files = [x for x in Path(fname).glob("*.csv")]
    if len(files) == 0:
        raise IOError("missing activity files .csv!")
    # files = [file.replace("\\", '/') for file in files]  # prevent Unix issues
    print(files)
    pool = Pool(processes=n_job)
    results = []
    for i, file in enumerate(files):
        results.append(
            pool.apply_async(read_activity_data, (file, i, len(files), window))
        )
    pool.close()
    pool.join()
    pool.terminate()

    data_first_sensor = pd.DataFrame()
    data_first_sensor_raw = pd.DataFrame()
    data_ss = pd.DataFrame()
    # data_second_sensor_min = pd.DataFrame()
    # data_second_sensor_max = pd.DataFrame()
    timestamp = None
    date_str = None
    print("preparing data for imputation...")
    for result in tqdm(results):
        a_data = result.get()
        activity = a_data[1]["first_sensor_value"]
        signal_strength = np.zeros(a_data[1].shape[0])
        if "signal_strength" in a_data[1]:
            signal_strength = a_data[1]["signal_strength"]

        x1 = np.zeros(a_data[1].shape[0])
        if "xmin" in a_data[1]:
            x1 = a_data[1]["xmin"]

        y1 = np.zeros(a_data[1].shape[0])
        if "xmax" in a_data[1]:
            y1 = a_data[1]["xmax"]

        z1 = np.zeros(a_data[1].shape[0])
        if "ymin" in a_data[1]:
            z1 = a_data[1]["ymin"]

        x2 = np.zeros(a_data[1].shape[0])
        if "ymax" in a_data[1]:
            x2 = a_data[1]["ymax"]

        y2 = np.zeros(a_data[1].shape[0])
        if "zmin" in a_data[1]:
            y2 = a_data[1]["zmin"]

        z2 = np.zeros(a_data[1].shape[0])
        if "zmax" in a_data[1]:
            z2 = a_data[1]["zmax"]

        power1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        power2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)

        nan_count = np.count_nonzero(np.isnan(activity.values))
        if abs(activity.size - nan_count) < 100:
            continue

        e = entropy_(activity)

        # activity[activity == 0] = np.nan
        if enable_remove_zeros:
            activity = a_data[1]["first_sensor_value"].replace(0, np.nan)
            # activity[activity < 2] = np.nan
        activity_o = activity.values
        power1_o = power1
        power2_o = power2

        if enable_anscombe:
            activity = anscombe(activity)

        if enable_log_anscombe:
            activity = np.log(anscombe(activity.values, 0))
            activity_reverse = np.array(
                [
                    x if np.isnan(x) else int(x)
                    for x in inverse_anscombe(np.exp(activity), 0)
                ]
            )  # cast to count

            assert np.array_equal(
                activity_o, activity_reverse, equal_nan=True
            ), "Inverse ascombe failed!"

            # power1 = np.log(anscombe_m(power1))
            # power2 = np.log(anscombe_m(power2))

        data_first_sensor[a_data[0]] = [e] + activity.tolist()

        data_first_sensor_raw[a_data[0]] = [e] + activity.tolist()

        data_ss[a_data[0]] = [e] + signal_strength.tolist()

        # xmin = a_data[1]["xmin"]
        # ymin = a_data[1]["ymin"]
        # zmin = a_data[1]["zmin"]
        # xmax = a_data[1]["xmax"]
        # ymax = a_data[1]["ymax"]
        # zmax = a_data[1]["zmax"]
        #
        # magnitude_min = math.sqrt(xmin * xmin + ymin * ymin + zmin * zmin)
        # magnitude_max = math.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)
        # data_second_sensor_min[a_data[0]] = magnitude_min
        # data_second_sensor_max[a_data[0]] = magnitude_max

        timestamp = a_data[1]["timestamp"]
        date_str = a_data[1]["date_str"]

        # d = []
        # thresh_pos = 800
        # s = np.array_split(activity, activity.shape[0] / 1440, axis=0)
        # for ii, x in enumerate(s):
        #     x = x.flatten()
        #     pos_count = x[x > 0].shape[0]
        #     if pos_count < thresh_pos:
        #         continue
        #     d.append(x)
        # if len(d) == 0:
        #     continue
        # tid = a_data[0]
        # vstack_transp = np.vstack(d)
        # df_ = pd.DataFrame(vstack_transp)
        # fig = go.Figure(data=go.Heatmap(
        #     z=df_.values,
        #     x=np.array(list(range(df_.values.shape[1]))),
        #     y=np.array(list(range(df_.values.shape[0]))),
        #     colorscale='Viridis'))
        # fig.update_layout(
        #     title="%dthresh=%s" % (tid, thresh_pos),
        #     xaxis_title="Time (1 min bins)",
        #     yaxis_title="Transponders")
        # filename = "%s_%d.html" % (tid, thresh_pos)
        # print(filename)
        # fig.write_html(filename)

    # data_first_sensor = data_first_sensor.dropna(axis=1, thresh=1000, how="any")
    # data_first_sensor = data_first_sensor.sort_values(data_first_sensor.first_valid_index(), axis=1, ascending=False)
    # data_ss = data_ss.sort_values(data_ss.first_valid_index(), axis=1, ascending=False)
    data_first_sensor = data_first_sensor.iloc[1:-1]
    data_ss = data_ss.iloc[1:-1]
    data_first_sensor_o = data_first_sensor.copy()
    # if n_top_traces > 0:
    #     data_first_sensor = data_first_sensor.iloc[:, : n_top_traces]
    #     data_ss = data_ss.iloc[:, : n_top_traces]
    print(data_first_sensor)

    # data_first_sensor = data_first_sensor.fillna(-1)
    # data_first_sensor_raw = data_first_sensor_raw.dropna(axis=1, thresh=1000, how="any")
    # data_first_sensor_raw = data_first_sensor_raw.sort_values(data_first_sensor_raw.first_valid_index(), axis=1,
    #                                                           ascending=False)
    data_first_sensor_raw = data_first_sensor_raw.iloc[1:-1]
    # if n_top_traces > 0:
    #     data_first_sensor_raw = data_first_sensor_raw.iloc[:, : n_top_traces]
    # data_first_sensor = data_first_sensor.fillna(-1)
    # data_m = data_first_sensor.notnull().astype(int).values
    data_x = data_first_sensor.astype(float).values
    data_x_raw = data_first_sensor_raw.astype(float).values
    ids = data_first_sensor.columns
    data_ss = data_ss.astype(float).values

    week_slice = []
    x = list(range(int(data_first_sensor.shape[0] / 1440)))
    for i in range(0, len(x), 7):
        slice_item = slice(i, i + 7, 1)
        week_slice.append(x[slice_item])
    n_week = len([x for x in week_slice if len(x) == 7])
    crop = n_week * 7 * 1440  # crop into 7 days chunck!

    return (
        data_x_raw[:crop, :],
        data_x[:crop, :],
        ids,
        timestamp[:crop],
        date_str[:crop],
        data_ss[:crop],
    )
    # return data_x_raw, data_x, ids, timestamp, date_str


def process(data_x, miss_rate):
    # Parameters
    no, dim = data_x.shape
    if miss_rate == 0:
        miss_data_x = data_x.copy()
        data_m = np.ones((no, dim), dtype=int)
        data_m[np.isnan(miss_data_x)] = 0
    else:
        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim, seed=0).astype(np.float)
        miss_data_x = data_x.copy()
        data_m[data_m == 0] = np.nan
        data_m[np.isnan(data_x)] = np.nan
        data_m[data_x == 0] = np.nan
        data_m[data_x == np.log(anscombe(0))] = np.nan
        miss_data_x[data_m == 1] = np.nan

    return miss_data_x, data_m

    # # Parameters
    # no, dim = data_x.shape
    #
    # # Introduce missing data
    # data_m = binary_sampler(1 - miss_rate, no, dim)
    # miss_data_x = data_x.copy()
    # miss_data_x[data_m == 0] = np.nan
    #
    # return miss_data_x, data_m


def main(args, raw_data, original_data_x, ids, timestamp, date_str, ss_data):
    """Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    """

    batch_size = args.batch_size
    hint_rate = args.hint_rate
    alpha = args.alpha
    iterations = args.iterations
    miss_rate = args.miss_rate
    output_dir = args.output_dir
    enable_anscombe = args.enable_anscombe
    n_top_traces = len(ids)
    reshape = args.reshape
    add_t_col = args.add_t_col
    thresh_daytime = args.thresh_daytime
    thresh_nan_ratio = args.thresh_nan_ratio
    export_csv = args.export_csv
    export_traces = args.export_traces
    export_heatmaps = args.export_heatmaps

    gain_parameters = {
        "batch_size": batch_size,
        "hint_rate": hint_rate,
        "alpha": alpha,
        "iterations": iterations,
    }
    RESHAPE = reshape.lower() in ["yes", "y", "t", "true"]
    ADD_TRANSP_COL = add_t_col.lower() in ["yes", "y", "t", "true"]
    N_TRANSPOND = int(n_top_traces)
    THRESH_DT = int(thresh_daytime)
    THRESH_NAN_R = int(thresh_nan_ratio)

    # Load data and introduce missingness
    data_x = original_data_x.copy()

    miss_data_x, data_m_x = process(data_x.copy(), miss_rate)
    imputed_data_x_li = linear_interpolation_v(miss_data_x.copy())

    # out = Path(output_dir) / "miss_rate_" + str(np.round(miss_rate, 4)).replace(".", "_") + "_iteration_" + \
    #       '%04d' % int(iterations) + "_thresh_" + str(THRESH_DT).replace(".", "_") + "_anscombe_" + str(
    #     enable_anscombe) + "_n_top_traces_" + str(n_top_traces)

    days = args.window_size
    out = (
        Path(output_dir)
        / f"{days}_miss_rate_{int(miss_rate * 100):04}_iteration_{int(iterations):04}_thresh_{THRESH_DT}_anscombe_{enable_anscombe}_n_top_traces_{n_top_traces}"
    )

    Path(out).mkdir(parents=True, exist_ok=True)

    if RESHAPE:
        (
            miss_data_x_reshaped_thresh,
            ss_reshaped_thresh,
            rm_row_idx,
            shape_o,
            transp_idx,
            activity_reshaped,
            ss_reshaped,
        ) = reshape_matrix_v1(
            ids,
            THRESH_NAN_R,
            ss_data,
            miss_data_x,
            timestamp,
            N_TRANSPOND,
            add_t_col=ADD_TRANSP_COL,
            thresh=THRESH_DT * days,
            days=days,
        )
    else:
        miss_data_x_reshaped_thresh = reshape_matrix_v2(miss_data_x)

    print(miss_data_x_reshaped_thresh)
    df = pd.DataFrame(miss_data_x_reshaped_thresh)
    header = [str(x) for x in range(miss_data_x_reshaped_thresh.shape[1])]
    for v in range(1, n_top_traces + 1):
        header[-v] = "t_%d" % (n_top_traces - v)
    header[-n_top_traces - 1] = "id"
    header[-n_top_traces - 2] = "epoch"
    df.columns = header
    dfs_transponder = [g for _, g in df.groupby(["id"])]

    header = [str(x) for x in range(ss_reshaped.shape[1])]
    header[-1] = "id"
    df_ss = pd.DataFrame(ss_reshaped)
    df_ss.columns = header
    dfs_ss = [g for _, g in df_ss.groupby(["id"])]

    pool = Pool(processes=args.n_job)
    for i in range(len(dfs_transponder)):
        result = pool.apply_async(
            worker_export_heatmaps,
            (
                i,
                len(dfs_transponder),
                dfs_transponder[i],
                n_top_traces,
                N_TRANSPOND,
                ss_reshaped[:, : -N_TRANSPOND - 1],
                dfs_ss[i].iloc[:, : -n_top_traces - 2],
                timestamp[0],
                THRESH_DT,
                out,
                miss_rate,
                export_heatmaps
            ),
        )

    xaxix_label, yaxis_label = result.get()[0], result.get()[1]
    pool.close()
    pool.join()
    pool.terminate()

    # for i in range(len(dfs_transponder)):
    #     d_t = dfs_transponder[i].iloc[:, :-n_top_traces - 2]
    #     valid = np.sum((~np.isnan(d_t.values)).astype(int))
    #     if valid <= 0:
    #         continue
    #     ss = ss_reshaped[:, :-N_TRANSPOND - 1]
    #     ss_t = dfs_ss[i].iloc[:, :-n_top_traces-2]
    #
    #     id = int(dfs_transponder[i]["id"].values[0])
    #     xaxix_label, yaxis_label = build_formated_axis(timestamp[0], min_in_row=d_t.shape[1], days_in_col=d_t.shape[0])
    #     xaxix_label[0] = xaxix_label[0].replace(minute=0)
    #     fig = go.Figure(data=go.Heatmap(
    #         z=d_t,
    #         x=xaxix_label,
    #         y=yaxis_label,
    #         colorscale='Viridis'))
    #     fig.update_xaxes(tickformat="%H:%M")
    #     fig.update_yaxes(tickformat="%d %b %Y")
    #     fig.update_layout(
    #         title="Transponder id %d thresh=%d" % (id, THRESH_DT),
    #         yaxis_title="Date(day)",
    #         xaxis_title="Time (1 min bins)")
    #     filename = out / f"{id}_reshaped_{THRESH_DT}_{valid}_filtered.html"
    #     #if i % 100 == 0:
    #     print(filename)
    #     fig.write_html(filename)

    # d_t_na = d_t.dropna(how="all")
    # fig = go.Figure(data=go.Heatmap(
    #     z=d_t_na,
    #     x=xaxix_label,
    #     y=np.arange(0, d_t_na.shape[1]),
    #     colorscale='Viridis'))
    # fig.update_xaxes(tickformat="%H:%M")
    # fig.update_yaxes(tickformat="%d %b %Y")
    # fig.update_layout(
    #     title="Transponder id %d thresh=%d" % (id, THRESH_DT),
    #     yaxis_title="Samples",
    #     xaxis_title="Time (1 min bins)")
    # filename = out / f"{id}_reshaped_{THRESH_DT}_{valid}_filtered.html"
    # #if i % 100 == 0:
    # print(filename)
    # fig.write_html(filename)

    # d_t_na_li = d_t.interpolate(method='linear', limit_direction='both', axis=1)
    # fig = go.Figure(data=go.Heatmap(
    #     z=d_t_na_li,
    #     x=xaxix_label,
    #     y=np.arange(0, d_t_na_li.shape[1]),
    #     colorscale='Viridis'))
    # fig.update_xaxes(tickformat="%H:%M")
    # fig.update_yaxes(tickformat="%d %b %Y")
    # fig.update_layout(
    #     title="Transponder id %d thresh=%d" % (id, THRESH_DT),
    #     yaxis_title="Samples",
    #     xaxis_title="Time (1 min bins)")
    # filename = out / f"{id}_reshaped_{THRESH_DT}_{valid}_li.html"
    # #if i % 100 == 0:
    # print(filename)
    # fig.write_html(filename)

    # fig = go.Figure(data=go.Heatmap(
    #     z=ss_t,
    #     x=xaxix_label,
    #     y=yaxis_label,
    #     colorscale='Viridis'))
    # fig.update_xaxes(tickformat="%H:%M")
    # fig.update_yaxes(tickformat="%d %b %Y")
    # fig.update_layout(
    #     title="%d Signal Strength thresh=%d" % (id, THRESH_DT),
    #     xaxis_title="Time (1 min bins)")
    # filename = out + "/" + "%d_signal_strength_reshaped_%d_%d.html" % (id, THRESH_DT, valid)
    # print(filename)
    # fig.write_html(filename)

    # if THRESH_DT > 0:
    #     fig = go.Figure(data=go.Heatmap(
    #         z=linear_interpolation_h(ss_t.values),
    #         x=xaxix_label,
    #         y=yaxis_label,
    #         colorscale='Viridis'))
    #     fig.update_xaxes(tickformat="%H:%M")
    #     fig.update_yaxes(tickformat="%d %b %Y")
    #     fig.update_layout(
    #         title="%d Signal Strength linear interpolated (row) thresh=%d" % (id, THRESH_DT),
    #         xaxis_title="Time (1 min bins)")
    #     filename = out + "/" + "%d_signal_strength_reshaped_ll_%d_%d.html" % (id, THRESH_DT, valid)
    #     print(filename)
    #     fig.write_html(filename)

    m = miss_data_x_reshaped_thresh[:, : -N_TRANSPOND - 1 - 1]
    # fig = go.Figure(data=go.Heatmap(
    #     z=m,
    #     x=xaxix_label,
    #     y=np.array(list(range(m.shape[0]))),
    #     colorscale='Viridis'))
    # fig.update_xaxes(tickformat="%H:%M")
    # fig.update_layout(
    #     title="Activity data 1 min bins thresh=%s" % THRESH_DT,
    #     xaxis_title="Time (1 min bins)",
    #     yaxis_title="Transponders")
    # filename = out / f"input_reshaped_{THRESH_DT}.html"
    # #if i % 100 == 0:
    # fig.write_html(filename)

    imputed_data_x, rmse_iter, rm_row_idx = gain(
        n_top_traces,
        xaxix_label,
        timestamp[0],
        miss_rate,
        out,
        THRESH_DT,
        ids,
        transp_idx,
        output_dir,
        shape_o,
        rm_row_idx,
        data_m_x.copy(),
        imputed_data_x_li.copy(),
        data_x.copy(),
        miss_data_x_reshaped_thresh.copy(),
        gain_parameters,
        out,
        RESHAPE,
        ADD_TRANSP_COL,
        N_TRANSPOND,
        days,
        args.n_job,
        export_heatmaps
    )

    del dfs_transponder
    del df

    # fig = go.Figure(data=go.Heatmap(
    #     z=imputed_data_x.T,
    #     x=np.array(list(range(imputed_data_x.shape[1]))),
    #     y=np.array(list(range(imputed_data_x.shape[0]))),
    #     colorscale='Viridis'))
    # filename = output_dir + "/" + "herd_gain_restored_%d.html" % 0
    # fig.write_html(filename)

    if export_csv:
        if miss_rate == 0:
            export_imputed_data(
                out,
                data_m_x,
                data_x,
                imputed_data_x,
                imputed_data_x_li,
                timestamp,
                date_str,
                ids,
                alpha,
                hint_rate,
                args.n_job,
            )

    # if export_traces:
    #   plot_imputed_data(out, imputed_data_x, imputed_data_x_li, raw_data, original_data_x, ids, timestamp)


def start(args):
    print(args)
    data_x_o, ori_data_x, ids, timestamp, date_str, ss_data = load_farm_data(
        args.data_dir,
        args.n_job,
        args.n_top_traces,
        enable_anscombe=args.enable_anscombe,
        enable_remove_zeros=args.enable_remove_zeros,
        enable_log_anscombe=args.enable_log_anscombe,
        window=args.window,
    )
    main(args, data_x_o, ori_data_x, ids, timestamp, date_str, ss_data)


def local_run(
    input_dir="F:/Data2/backfill_1min_cedara_fixed",
    output_dir="E:/thesis/gain/cedara",
    run_exp=False,
    n_top_traces=20,
    n_job=6,
    interation=100,
    export_heatmaps=False
):
    thresh_daytime = 100
    thresh_nan_ratio = 80

    if run_exp:
        for day in [1, 2, 3, 4, 5, 6, 7]:
            for miss_rate in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                # for n_traces in [10, 20, 30]:
                arg_run(
                    data_dir=input_dir,
                    output_dir=output_dir,
                    thresh_daytime=thresh_daytime,
                    thresh_nan_ratio=thresh_nan_ratio,
                    miss_rate=miss_rate,
                    n_top_traces=n_top_traces,
                    window_size=day,
                    n_job=n_job,
                    interation=interation,
                    export_heatmaps=export_heatmaps
                )
    else:
        arg_run(
            data_dir=input_dir,
            output_dir=output_dir,
            thresh_daytime=thresh_daytime,
            thresh_nan_ratio=thresh_nan_ratio,
            miss_rate=0,
            n_top_traces=60,
            n_job=n_job,
            window_size=1,
            interation=interation,
            export_heatmaps=export_heatmaps
        )

    # for miss_rate in np.arange(0.1, 0.99, 0.05):
    #     arg_run("F:/Data2/backfill_1min_delmas_fixed/delmas_70101200027", "E:/thesis/gain/delmas", thresh_daytime, thresh_nan_ratio, miss_rate)


def arg_run(
    data_dir=None,
    output_dir=None,
    thresh_daytime=100,
    thresh_nan_ratio=80,
    miss_rate=0,
    n_top_traces=-1,
    window_size=1,
    n_job=6,
    interation=20,
    export_heatmaps=False,
    output_hpc_string=False
):
    if output_hpc_string: #cores on BC4
        n_job = 20
        output_dir = f"/user/work/fo18103{output_dir.split(':')[1]}"
        data_dir = f"/user/work/fo18103{data_dir.split(':')[1]}"
    parser = argparse.ArgumentParser()
    if data_dir is None:
        parser.add_argument("data_dir", type=str)
        parser.add_argument("output_dir", type=str)
    else:
        parser.add_argument("--data_dir", type=str, default=data_dir)
        parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument(
        "--batch_size",
        help="the number of samples in mini-batch",
        default=128,
        type=int,
    )
    parser.add_argument("--hint_rate", help="hint probability", default=0.9, type=float)
    parser.add_argument("--alpha", help="hyperparameter", default=100, type=float)
    parser.add_argument(
        "--iterations",
        help="number of training interations",
        default=interation,
        type=int,
    )
    parser.add_argument(
        "--miss_rate", help="missing data probability", default=miss_rate, type=float
    )
    parser.add_argument(
        "--n_job", type=int, default=n_job, help="Number of thread to use."
    )
    parser.add_argument(
        "--window_size", type=int, default=window_size, help="Sample window size"
    )
    parser.add_argument(
        "--n_top_traces",
        type=int,
        default=n_top_traces,
        help="select n traces with highest entropy (<= 0 number to select all traces)",
    )
    parser.add_argument("--enable_anscombe", default=False, action='store_true')
    parser.add_argument("--enable_remove_zeros", default=False, action='store_true')
    parser.add_argument("--enable_log_anscombe", default=True, action='store_true')
    parser.add_argument("--window", default=False, action='store_true')
    parser.add_argument("--export_csv", default=True, action='store_true')
    parser.add_argument("--export_traces", default=True, action='store_true')
    parser.add_argument("--reshape", type=str, default="y")
    parser.add_argument("--w", type=str, default="y")
    parser.add_argument("--add_t_col", type=str, default="t")
    parser.add_argument("--thresh_daytime", type=str, default=thresh_daytime)
    parser.add_argument("--thresh_nan_ratio", type=str, default=thresh_nan_ratio)
    parser.add_argument("--export_heatmaps", default=export_heatmaps, action='store_true')

    args = parser.parse_args()
    if output_hpc_string:
        print_hpc_string(args)
        return
    start(args)


def print_hpc_string(args):
    hpc_s = "gain_imputation.py "
    for a in args._get_kwargs():
        if isinstance(a[1], bool):
            if a[1]:
                hpc_s += f" --{a[0]}"
            # else:
            #     hpc_s += f" --no-{a[0]}"
            continue
        hpc_s += f" --{a[0]} {a[1]}"
    print(f"'{hpc_s}'")
    with open('gain_hpc_ln.txt', 'a') as f:
        f.write(f"'{hpc_s}'" + "\n")
    with open('gain_hpc.txt', 'a') as f:
        f.write(f"'{hpc_s}'" + " ")


if __name__ == "__main__":
    arg_run()
    # local_run()

    # for n_top_traces in [20, 30, 40, 50]:
    #     local_run(input_dir="E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed", output_dir="E:/thesis/gain/cedara_exp",
    #               run_exp=True, interation=100, n_top_traces=n_top_traces)
    #
    #     local_run(input_dir="E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed", output_dir="E:/thesis/gain/delmas_exp",
    #               run_exp=True, interation=100, n_top_traces=n_top_traces)

    # # biospi
    # for n_top_traces in [20, 30, 40, 50]:
    #     local_run(
    #         input_dir="/mnt/storage/scratch/axel/thesis/activity_data/delmas/backfill_1min_delmas_fixed",
    #         output_dir="/mnt/storage/scratch/axel/thesis/gain/delmas",
    #         run_exp=True,
    #         n_top_traces=n_top_traces,
    #         n_job=20,
    #         interation=100
    #     )

    # local_run(
    #     input_dir="E:/thesis/activity_data/delmas/backfill_1min_delmas_fixed",
    #     output_dir="E:/thesis/gain/delmas",
    #     run_exp=True,
    #     n_top_traces=20,
    #     n_job=os.cpu_count()-2,
    #     interation=100
    # )

    # local_run(input_dir="E:/thesis/activity_data/cedara/backfill_1min_cedara_fixed", output_dir="E:/thesis/gain/cedara", run_exp=True)

    # local_run(input_dir="/mnt/storage/scratch/axel/gain/backfill_1min_cedara_fixed", output_dir="/mnt/storage/scratch/axel/gain/results/backfill_1min_cedara_fixed")
    # local_run(input_dir="/mnt/storage/scratch/axel/gain/backfill_1min_delmas_fixed", output_dir="/mnt/storage/scratch/axel/gain/results/backfill_1min_delmas_fixed")
