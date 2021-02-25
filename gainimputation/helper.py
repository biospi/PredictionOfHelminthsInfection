'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''

# Necessary packages
import warnings

import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.Utils import anscombe
import warnings
import pandas as pd


def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss(ori_data, imputed_data, imputed_data_li, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''

  ori_data_ = ori_data.copy()
  data_m[ori_data_ == np.log(anscombe(0))] = 1

  ori_data_norm, norm_parameters = normalization(ori_data)
  imputed_data_norm, _ = normalization(imputed_data, norm_parameters)
  imputed_data_li_norm, _ = normalization(imputed_data_li, norm_parameters)

  ori_data_norm[np.isnan(ori_data_norm)] = 0
  data_m[ori_data_norm == 0] = 1

  # Only for missing values
  original_masked = (1 - data_m) * ori_data_norm
  imputed_gain_masked = (1 - data_m) * imputed_data_norm
  imputed_li_masked = (1 - data_m) * imputed_data_li_norm

  # if original_masked[(imputed_gain_masked == 0) & (original_masked > 0)].size > 0:
  #   warnings.warn("erroneous point! nan or zeros in imputation results")
  #   original_masked[(imputed_gain_masked == 0) & (original_masked > 0)] = 0
  #
  # if original_masked[(original_masked == 0) & (imputed_gain_masked > 0)].size > 0:
  #   raise ValueError("erroneous point!")

  a = original_masked[original_masked > 0].size
  b = imputed_gain_masked[imputed_gain_masked > 0].size

  if a != b:
      print(a, b)
      raise ValueError("should have same number of point for rmse calculation!")

  diff_gain = original_masked - imputed_gain_masked
  nominator_gain = np.sum(diff_gain ** 2)
  denominator_gain = np.sum(1 - data_m)
  print("nominator gain=", nominator_gain)
  print("denominator gain=", denominator_gain)
  rmse_gain = np.sqrt(nominator_gain / float(denominator_gain))

  diff_li = original_masked - imputed_li_masked
  nominator_li = np.sum(diff_li ** 2)
  denominator_li = np.sum(1 - data_m)
  print("nominator li=", nominator_li)
  print("denominator li=", denominator_li)
  rmse_li = np.sqrt(nominator_li / float(denominator_li))

  return rmse_gain, rmse_li


def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev, seed=0)
      

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''

  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx


def rmse_loss_(ori_data, imputed_data, imputed_data_li, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    '''

    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    imputed_data_li, _ = normalization(imputed_data_li, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    nominator_ = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data_li) ** 2)
    denominator_ = np.sum(1 - data_m)

    rmse_ = np.sqrt(nominator_ / float(denominator_))

    return rmse, rmse_


def linear_interpolation(input_activity):
    for c in range(input_activity.shape[1]):
        i = np.array(input_activity[:, c], dtype=np.float)
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        input_activity[:, c] = s
    return input_activity

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