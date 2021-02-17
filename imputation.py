'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import math
from multiprocessing import Pool
import pandas as pd

import numpy as np
np.random.seed(0) #for reproducability

from gainimputation.gain import gain
from gainimputation.helper import binary_sampler
from gainimputation.helper import rmse_loss
from utils.Utils import create_rec_dir
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000000000
import matplotlib.dates as mdates
import datetime as dt
import scipy.stats
from utils.Utils import anscombe
import plotly.express as px
import json


def entropy_(to_resample):
    e = 0
    if to_resample.dropna().size > 0:
        e = scipy.stats.entropy(to_resample.dropna())
    return e


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def process_activity_data(file, i, nfiles, window):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")
    if window:
        w = 1440 * 3
        start = 413129
        end = start + w
        df_activity = df_activity.loc[start: end, :]

    # print(df_activity_w)
    # 411989 2015-11-04T02:29
    #159840

    return animal_id, df_activity


def plot_imputed_data(out, imputed_data_x_gain, imputed_data_x_li, raw_data, ori_data_x, ids, timestamps):
    print("plot_imputed_data...")
    out = out + "/figures/imp/"
    create_rec_dir(out)
    time_axis = np.array([dt.datetime.fromtimestamp(ts) for ts in timestamps])

    for i in range(imputed_data_x_li.shape[1]):


        imputed_li = imputed_data_x_li[:, i]
        imputed_gain = imputed_data_x_gain[:, i]
        original = ori_data_x[:, i]
        raw = raw_data[:, i]

        nan_count = np.count_nonzero(np.isnan(original))
        if nan_count == original.size or all(x == imputed_li[0] for x in imputed_li) \
                or all(x == original[0] for x in original):
            continue

        id = ids[i]
        date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(3)
        ax[1].xaxis_date()
        ax[1].xaxis.set_major_formatter(date_format)
        ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=60))
        ax[1].tick_params(axis='x', rotation=25)
        w = 500
        # ax[1].bar(time_axis[0:w], imputed_li[0:w], label="after li imputation", alpha=0.5, width=0.1)
        # ax[1].bar(time_axis[0:w], original[0:w], label="original", alpha=0.5, width=0.1)
        # ax[1].bar(time_axis[0:w], imputed_gain[0:w], label="after gain imputation", alpha=0.5, width=0.1)

        ax[1].plot(list(range(w)), imputed_li[0:w], label="after li imputation", alpha=0.5, marker='o')
        ax[1].plot(list(range(w)), original[0:w], label="original", alpha=0.5, marker='o')
        ax[1].plot(list(range(w)), imputed_gain[0:w], label="after gain imputation", alpha=0.5, marker='o')

        ax[1].set_title('Transformed activity before and after imputation')
        ax[1].legend()

        ax[2].plot(time_axis[0:w], np.abs(original[0:w] - imputed_gain[0:w]), label="original - gain", alpha=0.5, color='blue', linestyle='-')
        ax[2].plot(time_axis[0:w], np.abs(original[0:w] - imputed_li[0:w]), label="original - linear interp", alpha=0.5, color='red', linestyle='-')
        ax[2].legend()

        ax[0].plot(time_axis[0:w], raw[0:w], label="raw")
        # ax[0].set_title('Raw')

        rmse_gain = int(np.nansum(np.abs(original - imputed_gain)))
        rmse_li = int(np.nansum(np.abs(original - imputed_li)))
        fig.suptitle('gain %d      li %d' % (rmse_gain, rmse_li), fontsize=14)

        filename = "%d_gain_%d_li_%d.png" % (id, rmse_gain, rmse_li)
        filepath = "%s/%s" % (out, filename)
        # print('saving fig...')

        df = pd.DataFrame()
        df["time"] = time_axis.tolist() + time_axis.tolist() + time_axis.tolist()
        df["data"] = original.tolist() + imputed_li.tolist() + imputed_gain.tolist()
        df["imputation"] = ['original' for _ in range(len(original))] + ['linear interpolation' for _ in range(len(original))] + ['gain' for _ in range(len(original))]
        fig_px = px.bar(df, x="time", y="data", color='imputation', height=1340, barmode="group", title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li))

        fig_px.add_scatter(x=time_axis.tolist(), y=np.abs(original - imputed_li).tolist(), name="abs(original - imputed_li)", mode='lines+markers', marker=dict(color='coral'), connectgaps=True)
        fig_px.add_scatter(x=time_axis.tolist(), y=np.abs(original - imputed_gain).tolist(), name="abs(original - imputed_gain)", mode='lines+markers', marker=dict(color='green'), connectgaps=True)

        try:
            fig.tight_layout()
            fig.savefig(filepath)
            fig_px.write_html(filepath.replace(".png", ".html"))
        except OverflowError as e:
            print(e)
        plt.close(fig)
        # fig.show()
        # print("saved!")

        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(3)
        b = int(len(raw)/10)
        ax[0].hist(raw, bins=b, density=False)
        ax[0].set_title("Histogram of raw data " + str(id))
        ax[1].hist(original, bins=b, density=False)
        ax[1].set_title("Histogram of anscombe of raw data " + str(id))
        ax[2].hist(imputed_li, bins=b, density=False)
        ax[2].set_title("Histogram of imputed data " + str(id))
        fig.tight_layout()
        filename = "hist_%d.png" % id
        filepath = "%s/%s" % (out, filename)
        # print('saving fig...')
        try:
            plt.savefig(filepath)
        except OverflowError as e:
            print(e)
        plt.close(fig)


def export_imputed_data(out, data_m_x, ori_data_x, idata, timestamp, date_str, ids, alpha, hint):
    print("exporting imputed data...")
    print(ids)
    for i in range(idata.shape[1]):
        print("progress %d/%d ..." % (i, len(ids)))
        df = pd.DataFrame()
        df["timestamp"] = timestamp.values
        df["date_str"] = date_str.values
        df["first_sensor_value"] = ori_data_x[:, i]
        df["first_sensor_value_gain"] = idata[:, i]
        df["missingness"] = data_m_x[:, i]
        # df["signal_strength"] = 0
        # df["battery_voltage"] = 0
        #
        # df["xmin"] = 0
        # df["xmax"] = 0
        # df["ymin"] = 0
        # df["ymax"] = 0
        # df["zmin"] = 0
        # df["zmax"] = 0

        id = str(ids[i])
        filename = id + ".csv"
        filepath = out + "/" + filename
        df.to_csv(filepath, sep=',', index=False)
        print(filepath)


def load_farm_data(fname, n_job, n_top_traces=0, enable_anscombe=False, enable_log_anscombe=False, enable_remove_zeros=False, window=False):
    print("load_farm_data...")
    files = glob.glob(fname+"/*.csv")
    if len(files) == 0:
        raise IOError("missing activity files .csv!")
    files = [file.replace("\\", '/') for file in files]#prevent Unix issues
    print(files)
    pool = Pool(processes=n_job)
    results = []
    for i, file in enumerate(files):
        results.append(pool.apply_async(process_activity_data, (file, i, len(files), window)))
    pool.close()
    pool.join()
    pool.terminate()

    data_first_sensor = pd.DataFrame()
    data_first_sensor_raw = pd.DataFrame()
    # data_second_sensor_min = pd.DataFrame()
    # data_second_sensor_max = pd.DataFrame()
    timestamp = None
    date_str = None
    for result in results:
        a_data = result.get()
        activity = a_data[1]["first_sensor_value"]

        nan_count = np.count_nonzero(np.isnan(activity.values))
        if abs(activity.size - nan_count) < 100:
            continue

        e = entropy_(activity)

        #activity[activity == 0] = np.nan
        if enable_remove_zeros:
            activity = a_data[1]["first_sensor_value"].replace(0, np.nan)
        activity_o = activity

        if enable_anscombe:
            anscombe_m = np.vectorize(anscombe)
            # activity = anscombe_m(np.log(activity, out=np.zeros_like(activity), where=(activity != 0)))
            # activity = np.log(activity, out=np.zeros_like(activity), where=(activity != 0))
            activity = anscombe_m(activity)

        if enable_log_anscombe:
            anscombe_m = np.vectorize(anscombe)
            activity = np.log(anscombe_m(activity))

        data_first_sensor[a_data[0]] = [e] + activity.tolist()

        data_first_sensor_raw[a_data[0]] = [e] + activity_o.tolist()

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
    # data_first_sensor = data_first_sensor.dropna(axis=1, thresh=1000, how="any")
    data_first_sensor = data_first_sensor.sort_values(data_first_sensor.first_valid_index(), axis=1, ascending=False)
    data_first_sensor = data_first_sensor.iloc[1:]
    if n_top_traces > 0:
        data_first_sensor = data_first_sensor.iloc[:, : n_top_traces]
    print(data_first_sensor)

    # data_first_sensor = data_first_sensor.fillna(-1)

    # data_first_sensor_raw = data_first_sensor_raw.dropna(axis=1, thresh=1000, how="any")
    data_first_sensor_raw = data_first_sensor_raw.sort_values(data_first_sensor_raw.first_valid_index(), axis=1, ascending=False)
    data_first_sensor_raw = data_first_sensor_raw.iloc[1:]
    if n_top_traces > 0:
        data_first_sensor_raw = data_first_sensor_raw.iloc[:, : n_top_traces]

    # data_first_sensor = data_first_sensor.fillna(-1)
    #data_m = data_first_sensor.notnull().astype(int).values
    data_x = data_first_sensor.values
    data_x_raw = data_first_sensor_raw.values
    ids = data_first_sensor.columns
    return data_x_raw, data_x, ids, timestamp, date_str


def linear_interpolation(input_activity):
    for c in range(input_activity.shape[1]):
        i = np.array(input_activity[:, c], dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        input_activity[:, c] = s
    return input_activity


def process(data_x, miss_rate):
    # Parameters
    no, dim = data_x.shape
    if miss_rate == 0:
        miss_data_x = data_x.copy()
        data_m = np.ones((no, dim), dtype=int)
        data_m[np.isnan(miss_data_x)] = 0
    else:
        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim)
        miss_data_x = data_x.copy()
        miss_data_x[(data_m == 0) & ~np.isnan(data_x)] = np.nan

        data_m2 = np.ones((no, dim), dtype=int)
        data_m2[(data_m == 0) & ~np.isnan(data_x)] = 0
        data_m = data_m2.copy()

    return data_x, miss_data_x, data_m


# def reshape_matrix(matrix, days):
#     print(matrix.shape)
#     split = np.array_split(matrix, days, axis=0)
#     hstack = np.hstack(split)
#     print(hstack.shape)
#     return hstack
#
#
# def restore_matrix(matrix, n_transponder):
#     split = np.array_split(matrix, matrix.shape[1]/n_transponder, axis=1)
#     vstack = np.vstack(split)
#     return vstack

def reshape_matrix(matrix, days):
    print(matrix.shape)
    split = np.array_split(matrix, days, axis=0)
    #filter empty days
    filtered = []
    idx = []
    for i, s in enumerate(split):
        if (s > 0).any(): #day does contain data
            filtered.append(s)
            idx.append(i)  # store location of day
            continue

    hstack = np.hstack(split)
    hstack_filtered = np.hstack(filtered)
    print(hstack.shape)
    print(hstack_filtered.shape)
    return hstack_filtered, idx


def restore_matrix(matrix, imputed, n_transponder, idx_, days):
    split = np.array_split(matrix, days, axis=0)
    split_imp = np.array_split(imputed, imputed.shape[1] / n_transponder, axis=1)
    for i, d in enumerate(idx_):
        split[d] = split_imp[i]

    vstack = np.vstack(split)
    return vstack

def main(args, raw_data, original_data_x, ids, timestamp, date_str):
  '''Main function for UCI letter and spam datasets.
  
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
  '''
  
  # data_name = args.data_name
  # miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  RESHAPE = args.reshape.lower() in ["yes", 'y', 't', 'true']
  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  ori_data_x = original_data_x.copy()[:-1, :]
  ori_data_x_o = ori_data_x.copy()
  raw_data = raw_data[:-1, :]
  timestamp = timestamp[:-1]
  date_str = date_str[:-1]

  data_x, miss_data_x, data_m_x = process(ori_data_x, args.miss_rate)
  out = args.output_dir + "/miss_rate_" + str(np.round(args.miss_rate, 4)).replace(".", "_") + "_iteration_" +\
        '%04d' % int(args.iterations) + "_hint_rate_" + str(args.hint_rate).replace(".", "_") + "_alpha_" +\
        str(args.alpha).replace(".", "_") + "_anscombe_" + str(args.enable_anscombe) + "_n_top_traces_" + str(args.n_top_traces)
  create_rec_dir(out)
  # Impute missing data
  days = int(miss_data_x.shape[0]/1440)/2
  miss_data_x_o = miss_data_x.copy()

  if RESHAPE:
    miss_data_x, idx_ = reshape_matrix(miss_data_x, days)

  imputed_data_x = gain(miss_data_x, gain_parameters, out)

  if RESHAPE:
    imputed_data_x = restore_matrix(data_x.copy(), imputed_data_x, args.n_top_traces, idx_, days)

  imputed_data_x_li = linear_interpolation(miss_data_x_o)

  if args.export_csv:
    export_imputed_data(out, data_m_x, ori_data_x_o, imputed_data_x_li, timestamp, date_str, ids, args.alpha, args.hint_rate)

  #Report the RMSE performance
  rmse = rmse_loss(ori_data_x.copy(), imputed_data_x.copy(), data_m_x)
  print('RMSE Performance: ' + str(np.round(rmse, 4)))


  rmse_li = rmse_loss(ori_data_x.copy(), imputed_data_x_li.copy(), data_m_x)
  print('RMSE LI Performance: ' + str(np.round(rmse_li, 4)))

  rmse_info = {"rmse": rmse, "rmse_li": rmse_li}
  with open(out + '/rmse.json', 'w') as f:
      json.dump(rmse_info, f)

  imputed_data_x[data_m_x == 0] = np.nan
  imputed_data_x_li[data_m_x == 0] = np.nan
  ori_data_x[data_m_x == 0] = np.nan
  if args.export_traces:
    plot_imputed_data(out, imputed_data_x, imputed_data_x_li, raw_data, ori_data_x, ids, timestamp)

  # rmse_per_id = {}
  # rmse_per_id_li = {}
  # for i in range(ori_data_x.shape[1]):
  #     rmse_ = rmse_loss(ori_data_x[:, i], imputed_data_x[:, i], data_m[:, i], miss_data_x[:, i])
  #     id = str(ids[i])
  #     rmse_per_id[id] = rmse_
  #     rmse_li_ = rmse_loss(ori_data_x[:, i], imputed_data_x_li[:, i], data_m[:, i], miss_data_x[:, i])
  #     rmse_per_id_li[id] = rmse_li_

  return imputed_data_x, rmse, rmse_li


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  parser.add_argument('output_dir', type=str)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=1000,
      type=int)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.0,
      type=float)
  parser.add_argument('--n_job', type=int, default=6, help='Number of thread to use.')
  parser.add_argument('--n_top_traces', type=int, default=17, help='select n traces with highest entropy (<= 0 number to select all traces)')
  parser.add_argument('--enable_anscombe', type=bool, default=False)
  parser.add_argument('--enable_remove_zeros', type=bool, default=False)
  parser.add_argument('--enable_log_anscombe', type=bool, default=True)
  parser.add_argument('--window', type=bool, default=False)
  parser.add_argument('--export_csv', type=bool, default=True)
  parser.add_argument('--export_traces', type=bool, default=True)
  parser.add_argument('--reshape', type=bool, default=True)
  parser.add_argument('--w', type=bool, default=False)

  args = parser.parse_args() 
  
  # Calls main function
  data_x_o, ori_data_x, ids, timestamp, date_str = load_farm_data(args.data_dir, args.n_job, args.n_top_traces,
                                                        enable_anscombe=args.enable_anscombe,
                                                        enable_remove_zeros=args.enable_remove_zeros,
                                                        enable_log_anscombe=args.enable_log_anscombe,
                                                                  window=args.window)
  imputed_data, rmse, rmse_li, rmse_per_id, rmse_per_id_li = main(args, data_x_o, ori_data_x, ids, timestamp, date_str)
  print(imputed_data)
