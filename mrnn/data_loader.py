"""Functions for data loading.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from mrnn.utils import MinMaxScaler, anscombe, plot_heatmap


def data_loader (file_name, seq_len, missing_rate, start_i, end_i, has_weather, streams, run_id, stride, remove_zeros=True):
  """Load complete data and introduce missingness.
  
  Args:
    - file_name: the location of file to be loaded
    - seq_len: sequence length
    - missing_rate: rate of missing data to be introduced
    
  Returns:
    - x: data with missing values
    - m: observation indicator (m=1: observe, m=0: missing)
    - t: time information (time difference between two measurments)
    - ori_x: original data without missing values (for evaluation)
  """
  
  # Load the dataset
  data = pd.read_csv(file_name)
  temperature = data["temperature"]
  humidity = data["humidity"]

  data = data.iloc[start_i:]
  #data = data.loc[:, data.isnull().mean() < .9]

  if remove_zeros:
    data = data.replace(0, np.nan)
    # Create Random Mask
    # rand_zero_one_mask = np.random.randint(50, size=data.shape)
    # rand_zero_one_mask[~pd.isnull(data)] = 1
    # rand_zero_one_mask[:, -1] = 1
    # rand_zero_one_mask[:, -2] = 1
    # rand_zero_one_mask[:, -3] = 1
    # rand_zero_one_mask[:, -4] = 1
    # rand_zero_one_mask = rand_zero_one_mask.astype(bool)
    # # Fill df with 0 where mask is 0
    # data = data.where(rand_zero_one_mask, 0)

    #data[rand_zero_one_mask == 0] = 0
    #data = data.where(rand_zero_one_mask == 0, 0)

  timestamp = data["timestamp"].values
  date_str = data["date_str"].values
  data = data.drop('timestamp', 1)
  data = data.drop('date_str', 1)
  # data_m = data.copy()
  # data_m = data_m.replace(0, np.nan)
  # data_m = data_m.dropna(axis=1, thresh=1440*4)
  #
  # data = data[data_m.columns]

  w_str = ""
  if not has_weather:
    data = data.drop(['humidity', 'temperature'], 1)
  else:
    has_humidity = "humidity" in data.columns
    has_temp = "temperature" in data.columns
    if has_humidity:
      w_str += "humidity_"
    if has_temp:
      w_str += "temperature_"

  if end_i < 0:
    data = data[start_i:-1]
  # else:
  #   data = data[start_i: end_i]
  #data = data.iloc[:, 0:5]

  data = data[streams]
  data["temperature"] = temperature
  data["humidity"] = humidity

  if not has_weather:
    data = data.drop(['humidity', 'temperature'], 1)

  # data_anscombe = anscombe(data.values)
  # data_log_anscombe = np.log(data_anscombe)
  # data = pd.DataFrame(data_log_anscombe, columns=data.columns)

  #data = data.loc[:, (data > 0).sum() > 400]
  #data = data.loc[:, (data == 0).mean() < .4]

  data_o = data.copy()


  data_mask = data.copy()
  data_mask = data_mask.fillna(0)
  for i, col in enumerate(data.columns):
    data[col] = data[col].sample(frac=1-missing_rate, random_state=i)
    data_mask[col] = data_mask[col].sample(frac=1 - missing_rate, random_state=i)

  data_mask = np.isnan(data_mask).astype(int).values
  data_mask[np.isnan(data_o)] = 0

  features = data.columns.values
  data = data.values
  #data = data.T

  # Reverse time order
  data = data[::-1]

  # Normalize the data
  data, norm_parameters = MinMaxScaler(data)

  # data_o = data.copy()
  #plot_heatmap([pd.DataFrame(data, columns=features)], file_name, seq_len, 0, 0, 0, 0, "input")

  # Parameters
  no, dim = data.shape
  #no = no - seq_len

  # Define original data
  ori_x = list()
  print("define original data...")
  for i in range(0, no, stride):
    start = i
    end = i + seq_len
    #print(start, end)
    temp_ori_x = data[start: end]
    if temp_ori_x.shape[0] != seq_len:
        continue
    ori_x = ori_x + [temp_ori_x]
    
  # Introduce missingness
  m = list()
  x = list()
  t = list()
  print(f"introduce missingness {missing_rate}...")
  for i in range(len(ori_x)):
    # m
    # temp_m = 1*(np.random.uniform(0, 1, [seq_len, dim]) > missing_rate)
    # m = m + [temp_m]

    m_ = ori_x[i].copy()
    temp_m = (~np.isnan(m_)).astype(int)
    m = m + [temp_m]

    # x
    temp_x = ori_x[i].copy()
    temp_x[np.where(temp_m == 0)] = np.nan
    x = x + [temp_x]
    # t
    temp_t = np.ones([seq_len, dim])
    for j in range(dim):
      for k in range(1, seq_len):
        if temp_m[k, j] == 0:
          temp_t[k, j] = temp_t[k-1, j] + 1
    t = t + [temp_t]
    
  # Convert into 3d numpy array
  x = np.asarray(x)
  m = np.asarray(m)
  t = np.asarray(t)

  ori_x = np.asarray(ori_x)  
  
  # Fill 0 to the missing values
  x = np.nan_to_num(x, 0)
  #batch_size = x.shape[0]

  #df = pd.DataFrame(data_mask, columns=features)
  #df.to_csv(f"{run_id}_mask.csv", index=False)

  x_ = x.copy()
  x_[x_ == 0] = np.nan
  x_mean = np.nanmean(x_, axis=(0))
  x_median = np.nanmedian(x_, axis=(0))

  return x, m, t, ori_x, data_o, features, norm_parameters, w_str, data_mask, timestamp, date_str
   