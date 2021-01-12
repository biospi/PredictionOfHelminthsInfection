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

from gainimputation.gain import gain
from gainimputation.helper import binary_sampler
from gainimputation.helper import rmse_loss
from utils.Utils import create_rec_dir
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000000
import matplotlib.dates as mdates
import datetime as dt
import scipy.stats
from utils.Utils import anscombe


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


def plot_imputed_data(out, imputed_data_x, ori_data_x, ids, timestamps):
    print("plot_imputed_data...")
    out = out + "/figures/"
    create_rec_dir(out)
    time_axis = np.array([dt.datetime.fromtimestamp(ts) for ts in timestamps])
    for i in range(imputed_data_x.shape[1]):

        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(figsize=(39.20, 10.80))
        imputed = imputed_data_x[:, i]
        original = ori_data_x[:, i]

        nan_count = np.count_nonzero(np.isnan(original))
        if nan_count == original.size or all(x == imputed[0] for x in imputed) \
                or all(x == original[0] for x in original):
            continue

        id = ids[i]
        date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=60))
        ax.tick_params(axis='x', rotation=25)
        plt.plot(time_axis, imputed, label="after gainimputation imputation", alpha=0.5)
        plt.plot(time_axis, original, label="original", alpha=0.5)
        plt.title(id)
        plt.legend()

        filename = "%d.png" % id
        filepath = "%s/%s" % (out, filename)
        # print('saving fig...')
        plt.savefig(filepath)
        plt.close(fig)
        # fig.show()
        # print("saved!")


def export_imputed_data(out, ori_data_x, idata, timestamp, date_str, ids, alpha, hint):
    print("exporting imputed data...")
    print(ids)
    for i in range(idata.shape[1]):
        print("progress %d/%d ..." % (i, len(ids)))
        df = pd.DataFrame()
        df["timestamp"] = timestamp.values
        df["date_str"] = date_str.values
        df["first_sensor_value"] = ori_data_x[:, i]
        df["first_sensor_value_gain"] = idata[:, i]
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


def load_farm_data(fname, n_job, n_top_traces=0, enable_anscombe=False, window=False):
    print("load_farm_data...")
    files = glob.glob(fname+"/*.csv")
    if len(files) == 0:
        raise IOError("missing activity files .csv! in %s" % args.activity_dir)
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

        if enable_anscombe:
            anscombe_m = np.vectorize(anscombe)
            activity = anscombe_m(np.log(activity, out=np.zeros_like(activity), where=(activity != 0)))

        data_first_sensor[a_data[0]] = [e] + activity.tolist()

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
    data_first_sensor = data_first_sensor.dropna(axis=1, thresh=1000, how="any")
    data_first_sensor = data_first_sensor.sort_values(data_first_sensor.first_valid_index(), axis=1, ascending=False)
    data_first_sensor = data_first_sensor.iloc[1:]
    if n_top_traces > 0:
        data_first_sensor = data_first_sensor.iloc[:, : n_top_traces]


    print(data_first_sensor)
    # data_first_sensor = data_first_sensor.fillna(-1)
    #data_m = data_first_sensor.notnull().astype(int).values
    data_x = data_first_sensor.values
    ids = data_first_sensor.columns
    return data_x, ids, timestamp, date_str


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
        data_m[np.isnan(miss_data_x)] = 0
        miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m


def main(args, ori_data_x, ids, timestamp, date_str):
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
  
  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)



  data_x, miss_data_x, data_m = process(ori_data_x, args.miss_rate)
  out = args.output_dir + "/miss_rate_" + str(np.round(args.miss_rate, 4)).replace(".", "_") + "_iteration_" +\
        str(args.iterations).replace(".", "_") + "_hint_rate_" + str(args.hint_rate).replace(".", "_") + "_alpha_" +\
        str(args.alpha).replace(".", "_") + "_anscombe_" + str(args.enable_anscombe) + "_n_top_traces_" + str(args.n_top_traces)
  create_rec_dir(out)
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters, out)

  if args.export_csv:
    export_imputed_data(out, ori_data_x, imputed_data_x, timestamp, date_str, ids, args.alpha, args.hint_rate)

  plot_imputed_data(out, imputed_data_x, ori_data_x, ids, timestamp)

  # Report the RMSE performance
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  print('RMSE Performance: ' + str(np.round(rmse, 4)))

  imputed_data_x_li = linear_interpolation(miss_data_x)
  rmse_li = rmse_loss(ori_data_x, imputed_data_x_li, data_m)
  print('RMSE LI Performance: ' + str(np.round(rmse_li, 4)))

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
  parser.add_argument('--n_job', type=int, default=2, help='Number of thread to use.')
  parser.add_argument('--n_top_traces', type=int, default=-17, help='select n traces with highest entropy (<= 0 number to select all traces)')
  parser.add_argument('--enable_anscombe', type=bool, default=True)
  parser.add_argument('--export_csv', type=bool, default=True)
  args = parser.parse_args() 
  
  # Calls main function
  ori_data_x, ids, timestamp, date_str = load_farm_data(args.data_dir, args.n_job, args.n_top_traces,
                                                        enable_anscombe=args.enable_anscombe)
  imputed_data, rmse, rmse_li = main(args, ori_data_x, ids, timestamp, date_str)
  print(imputed_data)
