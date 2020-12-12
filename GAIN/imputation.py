'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
from multiprocessing import Pool
import pandas as pd

import numpy as np

from gain.gain import gain
from gain.helper import binary_sampler
from gain.helper import rmse_loss
from utils.Utils import create_rec_dir
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import datetime as dt


def parse_animal_id(file):
    animal_id = int(file.split("/")[-1].replace(".csv", ""))
    return animal_id


def process_activity_data(file, i, nfiles, start, end):
    print("process_activity_data processing files %d/%d  ..." % (i, nfiles))
    animal_id = parse_animal_id(file)
    df_activity = pd.read_csv(file, sep=",")
    w = 1440 * 3
    start = 413129
    end = start + w
    df_activity_w = df_activity.loc[start: end, :]
    # print(df_activity_w)
    # 411989 2015-11-04T02:29
    #159840

    return animal_id, df_activity_w


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
        if nan_count == original.size:
            continue

        id = ids[i]
        date_format = mdates.DateFormatter('%d/%b/%Y %H:%M')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        ax.tick_params(axis='x', rotation=25)
        plt.plot(time_axis, imputed, label="after gain imputation", alpha=0.5)
        plt.plot(time_axis, original, label="original", alpha=0.5)
        plt.title(id)
        plt.legend()

        filename = "%d.png" % id
        filepath = "%s/%s" % (out, filename)
        # print('saving fig...')
        plt.savefig(filepath)
        # print("saved!")


def export_imputed_data(out, ori_data_x, idata, timestamp, date_str, ids, alpha, hint):
    print("exporting imputed data...")
    create_rec_dir(out)
    print(ids)
    for i in range(idata.shape[1]):
        print("progress %d/%d ..." % (i, len(ids)))
        df = pd.DataFrame()
        df["timestamp"] = timestamp.values
        df["date_str"] = date_str.values
        df["first_sensor_value"] = ori_data_x[:, i]
        df["first_sensor_value_gain"] = idata[:, i]
        df["signal_strength"] = 0
        df["battery_voltage"] = 0

        df["xmin"] = 0
        df["xmax"] = 0
        df["ymin"] = 0
        df["ymax"] = 0
        df["zmin"] = 0
        df["zmax"] = 0

        id = str(ids[i])
        filename = id + ".csv"
        filepath = out + "/" + filename
        df.to_csv(filepath, sep=',', index=False)
        print(filepath)


def load_farm_data(fname, miss_rate, n_job):
    print("load_farm_data...")
    files = glob.glob(fname+"/*.csv")
    if len(files) == 0:
        raise IOError("missing activity files .csv! in %s" % args.activity_dir)
    files = [file.replace("\\", '/') for file in files]#prevent Unix issues
    print(files)
    pool = Pool(processes=n_job)
    results = []
    for i, file in enumerate(files):
        results.append(pool.apply_async(process_activity_data, (file, i, len(files), 0, 0)))
    pool.close()
    pool.join()
    pool.terminate()

    data = pd.DataFrame()
    timestamp = None
    date_str = None
    for result in results:
        a_data = result.get()
        data[a_data[0]] = a_data[1]["first_sensor_value"]
        timestamp = a_data[1]["timestamp"]
        date_str = a_data[1]["date_str"]
    print(data)
    data = data.fillna(-1)
    # data_m = data.notnull().astype(int).values
    data_x = data.values
    ids = data.columns

    # Parameters
    no, dim = data_x.shape
    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m, timestamp, date_str, ids



def main (args):
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

  ori_data_x, miss_data_x, data_m, timestamp, date_str, ids = load_farm_data(args.data_dir, args.miss_rate, args.n_job)

  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)
  export_imputed_data(args.output_dir, ori_data_x, imputed_data_x, timestamp, date_str, ids, args.alpha, args.hint_rate)
  # plot_imputed_data(args.output_dir, imputed_data_x, ori_data_x, ids, timestamp)
  # Report the RMSE performance
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

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
      default=10000,
      type=int)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument('--n_job', type=int, default=2, help='Number of thread to use.')
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data = main(args)
  print(imputed_data)
