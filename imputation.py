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
        w = 1440 * 7 * 12
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
    w = 1440 * 1
    start = 413129
    end = start + w
    time_axis = np.array([dt.datetime.fromtimestamp(ts) for ts in timestamps])[start:end]

    for i in range(imputed_data_x_li.shape[1]):


        imputed_li = imputed_data_x_li[:, i][start:end]
        imputed_gain = imputed_data_x_gain[:, i][start:end]
        original = ori_data_x[:, i][start:end]
        # raw = raw_data[:, i]

        nan_count = np.count_nonzero(np.isnan(original))
        if nan_count == original.size or all(x == imputed_li[0] for x in imputed_li) \
                or all(x == original[0] for x in original):
            continue

        id = ids[i]
        date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
        # plt.clf()
        # plt.cla()
        # fig, ax = plt.subplots(3)
        # ax[1].xaxis_date()
        # ax[1].xaxis.set_major_formatter(date_format)
        # ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=60))
        # ax[1].tick_params(axis='x', rotation=25)
        # w = 500
        # # ax[1].bar(time_axis[0:w], imputed_li[0:w], label="after li imputation", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], original[0:w], label="original", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], imputed_gain[0:w], label="after gain imputation", alpha=0.5, width=0.1)
        #
        # ax[1].plot(list(range(w)), imputed_li[0:w], label="after li imputation", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), original[0:w], label="original", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), imputed_gain[0:w], label="after gain imputation", alpha=0.5, marker='o')
        #
        # ax[1].set_title('Transformed activity before and after imputation')
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
        df["imputation"] = ['ORIGINAL' for _ in range(len(original))] + ['LI' for _ in range(len(original))] + ['GAIN' for _ in range(len(original))]
        color = []
        for ii in range(time_axis.size):
            o = original[ii]
            ili = imputed_li[ii]
            igain = imputed_gain[ii]
            if o == ili == igain:
                color.append("RAW")
                continue
            if np.isnan(o):
                color.append("TRUE_MISSING")
                continue
            color.append("ADDED_MISSING")

        df["color"] = color + color + color

        fig_px = px.bar(df, x="time", y="data", color='imputation', height=900, text='color', barmode="group", title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li))
        fig_px.update_traces(textposition='inside', insidetextanchor="start")
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
    data_first_sensor = data_first_sensor.iloc[1:-1]
    if n_top_traces > 0:
        data_first_sensor = data_first_sensor.iloc[:, : n_top_traces]
    print(data_first_sensor)

    # data_first_sensor = data_first_sensor.fillna(-1)
    # data_first_sensor_raw = data_first_sensor_raw.dropna(axis=1, thresh=1000, how="any")
    data_first_sensor_raw = data_first_sensor_raw.sort_values(data_first_sensor_raw.first_valid_index(), axis=1, ascending=False)
    data_first_sensor_raw = data_first_sensor_raw.iloc[1:-1]
    if n_top_traces > 0:
        data_first_sensor_raw = data_first_sensor_raw.iloc[:, : n_top_traces]
    # data_first_sensor = data_first_sensor.fillna(-1)
    #data_m = data_first_sensor.notnull().astype(int).values
    data_x = data_first_sensor.values
    data_x_raw = data_first_sensor_raw.values
    ids = data_first_sensor.columns

    week_slice = []
    x = list(range(int(data_first_sensor.shape[0]/1440)))
    for i in range(0, len(x), 7):
        slice_item = slice(i, i + 7, 1)
        week_slice.append(x[slice_item])
    n_week = len([x for x in week_slice if len(x) == 7])
    crop = n_week*7*1440

    return data_x_raw[:crop, :], data_x[:crop, :], ids, timestamp[:crop], date_str[:crop]


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


def linear_interpolation(input_activity):
    for c in range(input_activity.shape[1]):
        i = np.array(input_activity[:, c], dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        input_activity[:, c] = s
    return input_activity


def process(data_x_, miss_rate):
    # Parameters
    no, dim = data_x_.shape
    if miss_rate == 0:
        miss_data_x = data_x_.copy()
        data_m = np.ones((no, dim), dtype=int)
        data_m[np.isnan(miss_data_x)] = 0
    else:
        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim)
        miss_data_x = data_x_.copy()
        miss_data_x[(data_m == 0) & ~np.isnan(data_x_)] = np.nan

        data_m2 = np.ones((no, dim), dtype=int)
        data_m2[(data_m == 0) & ~np.isnan(data_x_)] = 0
        data_m = data_m2.copy()

    return miss_data_x, data_m


def reshape_matrix_ranjeet(matrix):
    print(matrix.shape)
    transp_block = []
    for i in range(matrix.shape[1]):
        transp = matrix[:, i]
        s = np.array_split(transp, matrix.shape[0]/1440, axis=0)
        s = [x.flatten() for x in s]
        vstack_transp = np.vstack(s)
        transp_block.append(vstack_transp)
    hstack = np.hstack(transp_block)
    return hstack


def restore_matrix_ranjeet(imputed, n_transpond):
    split = np.array_split(imputed, n_transpond, axis=1)
    matrix = []
    for s in split:
        days = []
        for i in range(s.shape[0]):
            d = s[i, :].reshape(-1, 1)
            days.append(d)
        vstack = np.vstack(days)
        matrix.append(vstack)
    hstack = np.hstack(matrix)
    return hstack


def reshape_matrix_andy(matrix, add_t_col=False, c=7):
    print(matrix.shape)
    
    transp_block = []
    for i in range(matrix.shape[1]):
        transp = matrix[:, i]
        s = np.array_split(transp, matrix.shape[0]/1440/c, axis=0)

        if add_t_col:
            d = []
            for ii, x in enumerate(s):
                x_ = x.flatten().tolist()
                x_d = x_ + [ii]
                d.append(np.array(x_d))
        else:
            d = [x.flatten() for x in s]

        vstack_transp = np.vstack(d)

        if add_t_col:
            df = pd.DataFrame(vstack_transp)
            for n in range(matrix.shape[1]):
                v = 1 if n == i else 0
                df["t_%d" % n] = v
            vstack_transp = df.values

        transp_block.append(vstack_transp)
    vstack = np.vstack(transp_block)
    return vstack


def restore_matrix_andy(imputed, n_transpond, add_t_col=False):
    if add_t_col:
        imputed = imputed[:, :-n_transpond-1] #-1 for date col
    split = np.array_split(imputed, n_transpond, axis=0)
    matrix = []
    for s in split:
        days = []
        for i in range(s.shape[0]):
            d = s[i, :].reshape(-1, 1)
            days.append(d)
        vstack = np.vstack(days)
        matrix.append(vstack)
    hstack = np.hstack(matrix)

    return hstack


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
  ADD_TRANSP_COL = args.add_t_col.lower() in ["yes", 'y', 't', 'true']

  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  # ori_data_x = original_data_x.copy()
  # ori_data_x_o = ori_data_x.copy()
  data_x = original_data_x.copy()
  miss_data_x, data_m_x = process(data_x, args.miss_rate)
  out = args.output_dir + "/miss_rate_" + str(np.round(args.miss_rate, 4)).replace(".", "_") + "_iteration_" +\
        '%04d' % int(args.iterations) + "_hint_rate_" + str(args.hint_rate).replace(".", "_") + "_alpha_" +\
        str(args.alpha).replace(".", "_") + "_anscombe_" + str(args.enable_anscombe) + "_n_top_traces_" + str(args.n_top_traces)
  create_rec_dir(out)

  miss_data_x_o = miss_data_x.copy()

  if RESHAPE:
    miss_data_x = reshape_matrix_andy(miss_data_x, add_t_col=ADD_TRANSP_COL)
  else:
    miss_data_x = reshape_matrix_ranjeet(miss_data_x)

    # if np.nansum(restore_matrix(miss_data_x, data_x.shape[1]) - miss_data_x_o) != 0:
    #     raise ValueError("Reshaping check failed!")

  imputed_data_x = gain(miss_data_x, gain_parameters, out)

  if np.isnan(imputed_data_x).all():
      raise ValueError("Error while imputing data, all value NaN!")

  if RESHAPE:
    imputed_data_x = restore_matrix_andy(imputed_data_x, data_x.shape[1], add_t_col=ADD_TRANSP_COL)
  else:
    imputed_data_x = restore_matrix_ranjeet(imputed_data_x, data_x.shape[1])

  imputed_data_x_li = linear_interpolation(miss_data_x_o.copy())

  if args.export_csv:
    export_imputed_data(out, data_m_x, data_x.copy(), imputed_data_x, timestamp, date_str, ids, args.alpha, args.hint_rate)

  #Report the RMSE performance
  rmse = rmse_loss(data_x.copy(), imputed_data_x.copy(), data_m_x)
  print('RMSE Performance: ' + str(np.round(rmse, 4)))


  rmse_li = rmse_loss(data_x.copy(), imputed_data_x_li.copy(), data_m_x)
  print('RMSE LI Performance: ' + str(np.round(rmse_li, 4)))

  rmse_info = {"rmse": rmse, "rmse_li": rmse_li}
  with open(out + '/rmse.json', 'w') as f:
      json.dump(rmse_info, f)

  # imputed_data_x[data_m_x == 0] = np.nan
  # imputed_data_x_li[data_m_x == 0] = np.nan
  # ori_data_x[data_m_x == 0] = np.nan
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
  parser.add_argument('--reshape', type=str, default='n')
  parser.add_argument('--w', type=str, default='n')
  parser.add_argument('--add_t_col', type=str, default='n')

  args = parser.parse_args() 
  
  # Calls main function
  data_x_o, ori_data_x, ids, timestamp, date_str = load_farm_data(args.data_dir, args.n_job, args.n_top_traces,
                                                        enable_anscombe=args.enable_anscombe,
                                                        enable_remove_zeros=args.enable_remove_zeros,
                                                        enable_log_anscombe=args.enable_log_anscombe,
                                                                  window=args.window)
  imputed_data, rmse, rmse_li, rmse_per_id, rmse_per_id_li = main(args, data_x_o, ori_data_x, ids, timestamp, date_str)
  print(imputed_data)
