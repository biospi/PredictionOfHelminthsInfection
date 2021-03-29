'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import glob
import math
from multiprocessing import Pool
import pandas as pd

import numpy as np

from data_imputation.data_loader import data_loader
from data_imputation.model_utils import imputation_performance
from data_imputation.mrnn import mrnn

np.random.seed(0) #for reproducability

from data_imputation.gain import gain
from data_imputation.helper import binary_sampler, linear_interpolation_v, reshape_matrix_andy, reshape_matrix_ranjeet, \
    restore_matrix_andy, build_formated_axis, linear_interpolation_h
from data_imputation.helper import rmse_loss
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
import plotly.graph_objects as go
from sys import exit


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
        print("WINDOW ON !!!")
        w = 1440 * 7 * 12
        start = 413129
        end = start + w
        df_activity = df_activity.loc[start: end, :]
    return animal_id, df_activity


def plot_imputed_data(out, imputed_data_x_gain, imputed_data_x_li, raw_data, ori_data_x, ids, timestamps):
    print("plot_imputed_data...")
    out = out + "/figures/imp/"
    create_rec_dir(out)
    w = 1440 * 3
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
        # # ax[1].bar(time_axis[0:w], imputed_li[0:w], label="after li data_imputation", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], original[0:w], label="original", alpha=0.5, width=0.1)
        # # ax[1].bar(time_axis[0:w], imputed_gain[0:w], label="after gain data_imputation", alpha=0.5, width=0.1)
        #
        # ax[1].plot(list(range(w)), imputed_li[0:w], label="after li data_imputation", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), original[0:w], label="original", alpha=0.5, marker='o')
        # ax[1].plot(list(range(w)), imputed_gain[0:w], label="after gain data_imputation", alpha=0.5, marker='o')
        #
        # ax[1].set_title('Transformed activity before and after data_imputation')
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
        df["data_imputation"] = ['ORIGINAL' for _ in range(len(original))] + ['LI' for _ in range(len(original))] + ['GAIN' for _ in range(len(original))]
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

        #fig_px = px.bar(df, x="time", y="data", color='data_imputation', height=900, text='color', barmode="group", title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li))
        fig_px = px.line(df, x="time", y="data", color='data_imputation', height=900, text='color', title="nominator rmse gain %d  rmse li %d" % (rmse_gain, rmse_li))
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


def export_imputed_data(out, data_m_x, ori_data_x, idata, timestamp, date_str, ids, alpha, hint):
    print("exporting imputed data...")
    print(ids)
    data_m_x[np.isnan(data_m_x)] = 1
    for i in range(idata.shape[1]):
        print("progress %d/%d ..." % (i, len(ids)))
        df = pd.DataFrame()
        df["timestamp"] = timestamp.values
        df["date_str"] = date_str.values
        df["first_sensor_value"] = ori_data_x[:, i]
        df["first_sensor_value_gain"] = idata[:, i]
        df["missingness"] = data_m_x[:, i]
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
    data_ss = pd.DataFrame()
    # data_second_sensor_min = pd.DataFrame()
    # data_second_sensor_max = pd.DataFrame()
    timestamp = None
    date_str = None
    for result in results:
        a_data = result.get()
        activity = a_data[1]["first_sensor_value"]
        signal_strength = a_data[1]["signal_strength"]

        x1 = a_data[1]["xmin"]
        y1 = a_data[1]["xmax"]
        z1 = a_data[1]["ymin"]
        x2 = a_data[1]["ymax"]
        y2 = a_data[1]["zmin"]
        z2 = a_data[1]["zmax"]

        power1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        power2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)

        nan_count = np.count_nonzero(np.isnan(activity.values))
        if abs(activity.size - nan_count) < 100:
            continue

        e = entropy_(activity)

        #activity[activity == 0] = np.nan
        if enable_remove_zeros:
            activity = a_data[1]["first_sensor_value"].replace(0, np.nan)
            #activity[activity < 2] = np.nan
        activity_o = activity
        power1_o = power1
        power2_o = power2

        if enable_anscombe:
            anscombe_m = np.vectorize(anscombe)
            # activity = anscombe_m(np.log(activity, out=np.zeros_like(activity), where=(activity != 0)))
            # activity = np.log(activity, out=np.zeros_like(activity), where=(activity != 0))
            activity = anscombe_m(activity)
            power1 = anscombe_m(power1)
            power2 = anscombe_m(power2)

        if enable_log_anscombe:
            anscombe_m = np.vectorize(anscombe)
            activity = np.log(anscombe_m(activity))
            power1 = np.log(anscombe_m(power1))
            power2 = np.log(anscombe_m(power2))

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
    data_first_sensor = data_first_sensor.sort_values(data_first_sensor.first_valid_index(), axis=1, ascending=False)
    data_ss = data_ss.sort_values(data_ss.first_valid_index(), axis=1, ascending=False)
    data_first_sensor = data_first_sensor.iloc[1:-1]
    data_ss = data_ss.iloc[1:-1]
    if n_top_traces > 0:
        data_first_sensor = data_first_sensor.iloc[:, : n_top_traces]
        data_ss = data_ss.iloc[:, : n_top_traces]
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
    data_ss = data_ss.values

    week_slice = []
    x = list(range(int(data_first_sensor.shape[0]/1440)))
    for i in range(0, len(x), 7):
        slice_item = slice(i, i + 7, 1)
        week_slice.append(x[slice_item])
    n_week = len([x for x in week_slice if len(x) == 7])
    crop = n_week*7*1440

    return data_x_raw[:crop, :], data_x[:crop, :], ids, timestamp[:crop], date_str[:crop], data_ss[:crop]
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
        data_m = binary_sampler(1 - miss_rate, no, dim)
        miss_data_x = data_x.copy()
        miss_data_x[(data_m == 0) & ~np.isnan(data_x) & (data_x != 0) & (data_x != np.log(anscombe(0)))] = np.nan

        data_m2 = np.ones((no, dim), dtype=int)
        data_m2[(data_m == 0) & ~np.isnan(data_x) & (data_x != 0) & (data_x != np.log(anscombe(0)))] = 0
        data_m = data_m2.copy()

    data_m = data_m.astype(np.float)
    data_m[data_m == 1] = np.nan
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
  N_TRANSPOND = int(args.n_top_traces)

  # Load data and introduce missingness
  data_x = original_data_x.copy()

  miss_data_x, data_m_x = process(data_x.copy(), args.miss_rate)
  imputed_data_x_li = linear_interpolation_v(miss_data_x.copy())

  thresh_pos = 500

  out = args.output_dir + "/miss_rate_" + str(np.round(args.miss_rate, 4)).replace(".", "_") + "_iteration_" +\
        '%04d' % int(args.iterations) + "_thresh_" + str(thresh_pos).replace(".", "_") + "_anscombe_" + str(args.enable_anscombe) + "_n_top_traces_" + str(args.n_top_traces)
  create_rec_dir(out)

  if RESHAPE:
    miss_data_x_reshaped, signal_s_reshaped, rm_row_idx, shape_o, transp_idx = reshape_matrix_andy(ss_data, miss_data_x, timestamp, N_TRANSPOND, add_t_col=ADD_TRANSP_COL, thresh=thresh_pos)
  else:
    miss_data_x_reshaped = reshape_matrix_ranjeet(miss_data_x)

  print(miss_data_x_reshaped)
  start = 0
  for i, k in enumerate(transp_idx):
      d = miss_data_x_reshaped[:, :-N_TRANSPOND-1]
      end = start+k
      d_t = d[start: end]
      ss = signal_s_reshaped[:, :-N_TRANSPOND - 1]
      ss_t = ss[start: end]

      start = end

      id = ids[i]
      xaxix_label, yaxis_label = build_formated_axis(timestamp[0], min_in_row=d_t.shape[1], days_in_col=d_t.shape[0])
      fig = go.Figure(data=go.Heatmap(
          z=d_t,
          x=xaxix_label,
          y=yaxis_label,
          colorscale='Viridis'))
      fig.update_xaxes(tickformat="%H:%M")
      fig.update_yaxes(tickformat="%d %b %Y")
      fig.update_layout(
          title="%d thresh=%d" % (id, thresh_pos),
          xaxis_title="Time (1 min bins)")
      filename = out + "/" + "%d_reshaped_%d.html" % (id, thresh_pos)
      print(filename)
      fig.write_html(filename)

      fig = go.Figure(data=go.Heatmap(
          z=ss_t,
          x=xaxix_label,
          y=yaxis_label,
          colorscale='Viridis'))
      fig.update_xaxes(tickformat="%H:%M")
      fig.update_yaxes(tickformat="%d %b %Y")
      fig.update_layout(
          title="%d Signal Strength thresh=%d" % (id, thresh_pos),
          xaxis_title="Time (1 min bins)")
      filename = out + "/" + "%d_signal_strength_reshaped_%d.html" % (id, thresh_pos)
      print(filename)
      fig.write_html(filename)

      fig = go.Figure(data=go.Heatmap(
          z=linear_interpolation_h(ss_t),
          x=xaxix_label,
          y=yaxis_label,
          colorscale='Viridis'))
      fig.update_xaxes(tickformat="%H:%M")
      fig.update_yaxes(tickformat="%d %b %Y")
      fig.update_layout(
          title="%d Signal Strength linear interpolated (row) thresh=%d" % (id, thresh_pos),
          xaxis_title="Time (1 min bins)")
      filename = out + "/" + "%d_signal_strength_reshaped_ll_%d.html" % (id, thresh_pos)
      print(filename)
      fig.write_html(filename)



  m = miss_data_x_reshaped[:, :-N_TRANSPOND-1]
  fig = go.Figure(data=go.Heatmap(
      z=m,
      x=xaxix_label,
      y=np.array(list(range(m.shape[0]))),
      colorscale='Viridis'))
  fig.update_xaxes(tickformat="%H:%M")
  fig.update_layout(
      title="Activity data 1 min bins thresh=%s" % thresh_pos,
      xaxis_title="Time (1 min bins)",
      yaxis_title="Transponders")
  filename = out + "/" + "input_reshaped_%d.html" % thresh_pos
  fig.write_html(filename)

  imputed_data_x, rmse_iter, rm_row_idx = gain(xaxix_label, timestamp[0], args.miss_rate, out, thresh_pos, ids, transp_idx, args.output_dir, shape_o, rm_row_idx, data_m_x.copy(), imputed_data_x_li.copy(), data_x.copy(), miss_data_x_reshaped.copy(), gain_parameters, out, RESHAPE, ADD_TRANSP_COL, N_TRANSPOND)

  # fig = go.Figure(data=go.Heatmap(
  #     z=imputed_data_x.T,
  #     x=np.array(list(range(imputed_data_x.shape[1]))),
  #     y=np.array(list(range(imputed_data_x.shape[0]))),
  #     colorscale='Viridis'))
  # filename = args.output_dir + "/" + "herd_gain_restored_%d.html" % 0
  # fig.write_html(filename)

  if args.export_csv:
    export_imputed_data(out, data_m_x, data_x, imputed_data_x, timestamp, date_str, ids, args.alpha, args.hint_rate)

  # if args.export_traces:
  #   plot_imputed_data(out, imputed_data_x, imputed_data_x_li, raw_data, original_data_x, ids, timestamp)


def mrnn_imputation(data, N_TRANSPOND, output_dir):
    # Inputs for the main function
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--seq_len',
    #     help='sequence length of time-series data',
    #     default=7,
    #     type=int)
    # parser.add_argument(
    #     '--missing_rate',
    #     help='the rate of introduced missingness',
    #     default=0.2,
    #     type=float)
    # parser.add_argument(
    #     '--h_dim',
    #     help='hidden state dimensions',
    #     default=10,
    #     type=int)
    # parser.add_argument(
    #     '--batch_size',
    #     help='the number of samples in mini batch',
    #     default=128,
    #     type=int)
    # parser.add_argument(
    #     '--iteration',
    #     help='the number of iteration',
    #     default=2000,
    #     type=int)
    # parser.add_argument(
    #     '--learning_rate',
    #     help='learning rate of model training',
    #     default=0.01,
    #     type=float)
    # parser.add_argument(
    #     '--metric_name',
    #     help='imputation performance metric',
    #     default='rmse',
    #     type=str)
    #
    # args = parser.parse_args()

    print("mrnn_imputation....")
    ## Load data
    x, m, t, ori_x = data_loader(data)
    # mrnn model parameters
    model_parameters = {'h_dim': 10,
                        'batch_size': 128,
                        'iteration': 2000,
                        'learning_rate': 0.01}
    # Fit mrnn_model
    mrnn_model = mrnn(x, model_parameters)
    print("fitting....")
    mrnn_model.fit(x, m, t)

    # Impute missing data
    imputed_x = mrnn_model.transform(x, m, t)

    fig = go.Figure(data=go.Heatmap(
        z=imputed_data[:, :-N_TRANSPOND - 1],
        x=np.array(list(range(imputed_data.shape[1]))[:-N_TRANSPOND - 1]),
        y=np.array(list(range(imputed_data.shape[0]))),
        colorscale='Viridis'))
    filename = output_dir + "/" + "imputed_%d.html" % 0
    fig.write_html(filename)

    # Evaluate the data_imputation performance
    performance = imputation_performance(ori_x, imputed_x, m, 'rmse')

    # Report the result
    print('rmse: ' + str(np.round(performance, 4)))


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
  data_x_o, ori_data_x, ids, timestamp, date_str, ss_data = load_farm_data(args.data_dir, args.n_job, args.n_top_traces,
                                                        enable_anscombe=args.enable_anscombe,
                                                        enable_remove_zeros=args.enable_remove_zeros,
                                                        enable_log_anscombe=args.enable_log_anscombe,
                                                                  window=args.window)
  imputed_data, rmse, rmse_li, rmse_per_id, rmse_per_id_li = main(args, data_x_o, ori_data_x, ids, timestamp, date_str, ss_data)
  print(imputed_data)
