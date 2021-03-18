'''data_imputation function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "data_imputation: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import json

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from data_imputation.helper import normalization, renormalization, rounding, rmse_loss, rmse_loss_, linear_interpolation
from data_imputation.helper import xavier_init, restore_matrix_andy, restore_matrix_ranjeet
from data_imputation.helper import binary_sampler, uniform_sampler, sample_batch_index

import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd


def gain(miss_rate, out, thresh, ids, t_idx, output_dir, shape_o, nan_row_idx, data_m_x, imputed_data_x_li, data_x_o, data_x, gain_parameters, outpath, RESHAPE, ADD_TRANSP_COL, N_TRANSPOND):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: data_imputation network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)

  # norm_data = data_x
  # norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## data_imputation architecture
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## data_imputation functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## data_imputation structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## data_imputation loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## data_imputation solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  rmse_gain = []
  rmse_li = []
  rmse_iter = []
  # Start Iterations
  i = 0
  range_iter = range(iterations)
  for it in tqdm(range_iter):
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    # imputed_data = rounding(imputed_data, data_x)


    if (i % 100 == 0) | (i == 0) | (i == range_iter[-1]):
    #if True:
      rmse_iter.append(i)
      i_d = imputed_data[:, :-N_TRANSPOND - 1]
      fig = go.Figure(data=go.Heatmap(
        z=i_d,
        x=np.array(list(range(i_d.shape[1]))),
        y=np.array(list(range(i_d.shape[0]))),
        colorscale='Viridis'))
      fig.update_layout(
        title="Activity data after imputation i=%s" % i,
        xaxis_title="Time (1 min bins)",
        yaxis_title="Transponders")

      filename = output_dir + "/" + "imputed_gain_%d.html" % i
      print(filename)
      fig.write_html(filename)

      df_ = pd.DataFrame(i_d)
      start = 0
      for n, item in enumerate(t_idx):
          end = start + item
          df_t_i = df_[start: end]
          start = end
          id = ids[n]
          fig = go.Figure(data=go.Heatmap(
            z=df_t_i.values,
            x=np.array(list(range(df_t_i.values.shape[1]))),
            y=np.array(list(range(df_t_i.values.shape[0]))),
            colorscale='Viridis'))
          fig.update_layout(
            title="imputed %d thresh=%d iteration=%d" % (id, thresh, i),
            xaxis_title="Time (1 min bins)",
            yaxis_title="Days")
          filename = out + "/" + "%d_imputed_reshaped_%d_%d.html" % (id, thresh, i)
          print(filename)
          fig.write_html(filename)


      '''
      
      RMSE Calculation
      
      '''

      if np.isnan(imputed_data).any():
        warnings.warn("Warning NaN in normalised imputed results.")

      if np.isnan(imputed_data).all():
        raise ValueError("Error while imputing data, all value NaN!")

      if RESHAPE:
        imputed_data = restore_matrix_andy(output_dir, shape_o, nan_row_idx, imputed_data, N_TRANSPOND, add_t_col=ADD_TRANSP_COL)
      else:
        imputed_data = restore_matrix_ranjeet(imputed_data, N_TRANSPOND)


      if miss_rate > 0:
        # rmse_g, rmse_l = rmse_loss(data_x_o.copy(), imputed_data.copy(), imputed_data_x_li.copy(), data_m_x, output_dir, i)
        rmse_g, rmse_l = rmse_loss(data_x_o.copy(), imputed_data.copy(), imputed_data_x_li.copy(), data_m_x, output_dir, i)

        print('RMSE GAIN Performance: ' + str(np.round(rmse_g, 4)))
        print('RMSE LI Performance: ' + str(np.round(rmse_l, 4)))
        rmse_gain.append(rmse_g)
        rmse_li.append(rmse_l)

        rmse_info = {"rmse": rmse_g, "rmse_li": rmse_l}
        with open(outpath + '/rmse_%i.json' % i, 'w') as f:
          json.dump(rmse_info, f)

    i += 1
  if miss_rate > 0:
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    ax.set_ylabel('RMSE')
    ax.set_xlabel('iteration')
    plt.plot(rmse_iter, rmse_gain, label="RMSE GAIN", alpha=1)
    plt.plot(rmse_iter, rmse_li, label="RMSE LI", alpha=1)

    plt.title("RMSE iteration performance")
    plt.legend()
    filename = outpath + "/" + "RMSE.png"
    print(filename)
    plt.savefig(filename)

  return imputed_data, rmse_iter