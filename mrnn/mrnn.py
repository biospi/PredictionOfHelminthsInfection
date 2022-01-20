"""MRNN core functions.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using
           Multi-Directional Recurrent Neural Networks,"
           in IEEE Transactions on Biomedical Engineering,
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) Train RNN part
(2) Test RNN part
(3) Train FC part
(4) Test FC part
"""

# Necessary Packages
import pathlib
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
from tensorflow import keras
from tensorflow.python.keras.backend import placeholder
from tqdm import tqdm
from mrnn.utils import plot_loss_curve, plot_model_struct, plot_heatmap_imstep, initialise_time_matrix, MinMaxScaler, \
  MinMaxScaler_, Denormalization
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from pathlib import Path
from mrnn.model_utils import biGRUCell, initial_point_interpolation
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()


class mrnn ():
  """MRNN class with core functions.

  Attributes:
    - x: incomplete data
    - model_parameters:
      - h_dim: hidden state dimensions
      - batch_size: the number of samples in mini-batch
      - iteration: the number of iteration
      - learning_rate: learning rate of model training
  """

  def __init__(self, n, x, model_parameters, missing_rate, file, weather_str,
               start_i, end_i, run_id, streams, x_train_shape, stride, seq_len):

    # Set Parameters
    self.no, self.seq_len, self.dim = x.shape
    self.missing_rate = missing_rate
    self.file = file
    self.weather_str = weather_str
    self.h_dim = model_parameters['h_dim']
    self.batch_size = model_parameters['batch_size']
    self.iteration = model_parameters['iteration']
    self.learning_rate = model_parameters['learning_rate']
    self.start_i = start_i
    self.end_i = end_i
    self.n = n
    self.run_id = run_id
    self.streams = streams
    self.x_train_shape = x_train_shape
    self.stride = stride
    self.seq_len = seq_len

  def rnn_train (self, x, m, t, f):
    """Train RNN for each feature.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
      - f: feature index
    """
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:

      # input place holders
      target = placeholder(dtype=tf.float32, shape=[self.seq_len, None, 1])
      mask = placeholder(dtype=tf.float32, shape=[self.seq_len, None, 1])

      # Build rnn object
      rnn = biGRUCell(3, self.h_dim, 1)
      outputs = rnn.get_outputs()
      loss = tf.sqrt(tf.reduce_mean(tf.square(mask*outputs - mask*target)))
      optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
      train = optimizer.minimize(loss)

      #sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.global_variables_initializer())

      # Training
      rnn_loss = []
      for i in range(self.iteration):
        # Batch selection
        batch_idx = np.random.permutation(x.shape[0])[:self.batch_size]

        temp_input = np.dstack((x[:,:,f], m[:,:,f], t[:,:,f]))
        temp_input_reverse = np.flip(temp_input, 1)

        forward_input = np.zeros([self.batch_size, self.seq_len, 3])
        forward_input[:,1:,:] = temp_input[batch_idx,
                                           :(self.seq_len-1), :]

        backward_input = np.zeros([self.batch_size, self.seq_len, 3])
        backward_input[:,1:,:] = temp_input_reverse[batch_idx,
                                                    :(self.seq_len-1),:]

        a, step_loss = \
        sess.run([train, loss],
                 feed_dict=
                 {mask: np.transpose(np.dstack(m[batch_idx,:,f]),[1, 2, 0]),
                  target: np.transpose(np.dstack(x[batch_idx,:,f]),[1, 2, 0]),
                  rnn._inputs: forward_input,
                  rnn._inputs_rev: backward_input})
        rnn_loss.append(step_loss)

      # Save model
      inputs = {'forward_input': rnn._inputs,
                'backward_input': rnn._inputs_rev}
      outputs = {'imputation': outputs}

      save_file_name = str(pathlib.Path(f"tmp/mrnn_imputation_{self.start_i}_{self.end_i}_{self.iteration}_{self.seq_len}_{self.n}_{self.run_id}/rnn_feature_" + str(f+1) + '/'))
      tf.compat.v1.saved_model.simple_save(sess, save_file_name,
                                           inputs, outputs)
      return rnn_loss


  def rnn_predict (self, x, m, t, train_test_label, i=0):
    """Impute missing data using RNN block.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix

    Returns:
      - imputed_x: imputed data by rnn block
    """

    # x_ = x.copy()
    # x_o = x.copy()
    # x_[x_ == 0] = np.nan
    # x_mean = np.nanmean(x_, axis=(0))
    # x_median = np.nanmedian(x_, axis=(0))
    # x_mean[np.isnan(x_mean)] = 0
    # x_median[np.isnan(np.nan)] = 0
    #
    # x_mean = np.mean(x, axis=(0))

    #print(x.shape)
    # plot_heatmap_imstep(self.streams, x, "0 raw input", train_test_label, self.run_id, self.missing_rate,
    #                     self.seq_len, self.iteration, self.weather_str, self.n, m=m, i=i)

    # Output Initialization
    imputed_x = np.zeros([self.no, self.seq_len, self.dim])

    # For each feature
    for f in range(self.dim):
      temp_input = np.dstack((x[:,:,f], m[:,:,f], t[:,:,f]))
      temp_input_reverse = np.flip(temp_input, 1)

      forward_input = np.zeros([self.no, self.seq_len, 3])
      forward_input[:,1:,:] = temp_input[:,:(self.seq_len-1), :]

      backward_input = np.zeros([self.no, self.seq_len, 3])
      backward_input[:,1:,:] = temp_input_reverse[:,:(self.seq_len-1),:]

      # import os
      # root = Path(os.path.dirname(os.path.dirname(__file__)))

      save_file_name = f"tmp/mrnn_imputation_{self.start_i}_{self.end_i}_{self.iteration}_{self.seq_len}_{self.n}_{self.run_id}/rnn_feature_" + str(f+1)  + '/'

      # filepath = root / 'mrnn' / save_file_name
      #
      # save_file_name = str(filepath)

      # Load saved model
      graph = tf.Graph()
      graph.as_default()
      with graph.as_default():
        with tf.compat.v1.Session() as sess:
          sess.run(tf.compat.v1.global_variables_initializer())
          tf.compat.v1.saved_model.loader.load(sess,
                                               [tf.compat.v1.saved_model.SERVING],
                                               save_file_name)
          fw_input = graph.get_tensor_by_name('inputs:0')
          bw_input = graph.get_tensor_by_name('inputs_rev:0')
          output = graph.get_tensor_by_name('map/TensorArrayV2Stack/TensorListStack:0')

          imputed_data = sess.run(output,
                                  feed_dict={fw_input: forward_input,
                                             bw_input: backward_input})

          # for i in range(x.shape[0]):
          #   for j in range(x.shape[1]):
          #     if np.all(m[i, j, :] == 0):
          #       x[i, j, :] = x_mean[j, :]
          #       m[i, j, :] = 1

          imputed_x[:, :, f] = (1-m[:,:,f]) * np.transpose(np.squeeze(imputed_data)) + \
                               m[:,:,f] * x[:,:,f]

    # Initial poitn interpolation for better performance
    #imputed_x = initial_point_interpolation(x, m, t, imputed_x, self.weather_str)

    #plot_heatmap_imstep(self.streams, imputed_x, "2 bi-rnn output after point interpolation", train_test_label, self.run_id, self.missing_rate, self.seq_len, self.iteration, self.weather_str, self.n, i=i)

    return imputed_x


  def fc_train(self, x, m, t):
    """Train Fully Connected Networks after RNN block.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
    """
    tf.compat.v1.reset_default_graph()

    # rnn imputation results
    rnn_imputed_x = self.rnn_predict(x, m, t, "training", 0)

    # Reshape the data for FC train
    x = np.reshape(x, [self.no * self.seq_len, self.dim])
    rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
    m = np.reshape(m, [self.no * self.seq_len, self.dim])

    # input place holders
    x_input = placeholder(dtype=tf.float32, shape=[None, self.dim])
    target = placeholder(dtype=tf.float32, shape=[None, self.dim])
    mask = placeholder(dtype=tf.float32, shape=[None, self.dim])

    # build a FC network
    U = tf.compat.v1.get_variable("U", shape=[self.dim, self.dim],
                                  initializer=tf.compat.v1.keras.initializers.glorot_normal)
    V1 = tf.compat.v1.get_variable("V1", shape=[self.dim, self.dim],
                                   initializer=tf.compat.v1.keras.initializers.glorot_normal)
    V2 = tf.compat.v1.get_variable("V2", shape=[self.dim, self.dim],
                                   initializer=tf.compat.v1.keras.initializers.glorot_normal)
    b = tf.Variable(tf.random.normal([self.dim]))

    L1 = tf.nn.sigmoid((tf.matmul(x_input, tf.linalg.set_diag(U, np.zeros([self.dim,]))) + \
                        tf.matmul(target, tf.linalg.set_diag(V1, np.zeros([self.dim,]))) + \
                        tf.matmul(mask, V2) + b))

    W = tf.Variable(tf.random.normal([self.dim]))
    a = tf.Variable(tf.random.normal([self.dim]))
    hypothesis = W * L1 + a

    outputs = tf.nn.sigmoid(hypothesis)

    # reshape out for sequence_loss
    loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - target)) )

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
    train = optimizer.minimize(loss)

    # Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Training step
    fc_loss = []
    for i in range(self.iteration * 20):
      batch_idx = np.random.permutation(x.shape[0])[:self.batch_size]
      a, step_loss = sess.run([train, loss],
                              feed_dict={x_input: x[batch_idx, :],
                                         target: rnn_imputed_x[batch_idx, :],
                                         mask: m[batch_idx, :]})
      fc_loss.append(step_loss)

    plot_loss_curve(self.run_id, self.streams, self.n, self.start_i, self.end_i, [fc_loss], self.file, 'Fully Connected Networks after RNN block training loss', self.seq_len, self.missing_rate, self.iteration, self.learning_rate, self.weather_str)

    # Save model
    inputs = {'x_input': x_input,
              'target': target,
              'mask': mask}
    outputs = {'imputation': outputs}

    save_file_name = str(pathlib.Path(f"tmp/mrnn_imputation_{self.start_i}_{self.end_i}_{self.iteration}_{self.seq_len}_{self.n}_{self.run_id}/fc_feature/"))
    tf.compat.v1.saved_model.simple_save(sess, save_file_name,
                                         inputs, outputs)

  def rnn_fc_predict(self, x, m, t, train_test_label, i=0):
    """Impute missing data using RNN and FC.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix

    Returns:
      - fc_imputed_x: imputed data using RNN and FC
    """
    # rnn imputation results
    rnn_imputed_x = self.rnn_predict(x, m, t, "testing", i)
    # Reshape the data for FC predict
    x = np.reshape(x, [self.no * self.seq_len, self.dim])
    rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
    m = np.reshape(m, [self.no * self.seq_len, self.dim])

    save_file_name = f"tmp/mrnn_imputation_{self.start_i}_{self.end_i}_{self.iteration}_{self.seq_len}_{self.n}_{self.run_id}/fc_feature/"

    # Load saved data
    graph = tf.Graph()
    graph.as_default()
    with graph.as_default():
      with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.SERVING],
                                             save_file_name)
        x_input = graph.get_tensor_by_name('Placeholder:0')
        target = graph.get_tensor_by_name('Placeholder_1:0')
        mask = graph.get_tensor_by_name('Placeholder_2:0')
        outputs = graph.get_tensor_by_name('Sigmoid_1:0')

        fc_imputed_x = sess.run(outputs, feed_dict={x_input: x,
                                                    target: rnn_imputed_x,
                                                    mask: m})

        # Reshape imputed data to 3d array
        fc_imputed_x = np.reshape(fc_imputed_x, [self.no, self.seq_len, self.dim])
        m = np.reshape(m, [self.no, self.seq_len, self.dim])
        x = np.reshape(x, [self.no, self.seq_len, self.dim])

        # x_ = x.copy()
        # x_[x_ == 0] = np.nan
        # x_mean = np.nanmean(x_, axis=(0))
        # x_median = np.nanmedian(x_, axis=(0))
        # x_mean[np.isnan(x_mean)] = 0
        # x_median[np.isnan(np.nan)] = 0
        #
        # X_mean = np.zeros(fc_imputed_x.shape)
        # X_median = np.zeros(fc_imputed_x.shape)
        # for k in range(X_mean.shape[0]):
        #     X_mean[k, :, :] = x_mean
        #     X_median[k, :, :] = x_median
        #
        # no, seq_len, dim = x.shape
        #
        m_ = m.copy()
        # for k in range(no):
        #     for j in range(seq_len):
        #         if np.all(m[k, j, :] == 0):
        #             x[k, j, :] = x_mean[j, :]
        #             m_[k, j, :] = np.ones(x_mean[j, :].shape)

        fc_imputed_x = fc_imputed_x * (1-m) + x * m_

        #plot_heatmap_imstep(self.streams, fc_imputed_x, "3 raw fully connected output", train_test_label, self.run_id, self.missing_rate, self.seq_len, self.iteration, self.weather_str, self.n, i=i)
        #fc_imputed_x = initial_point_interpolation(x, m, t, fc_imputed_x, self.weather_str)

        x_ = x.copy()
        x_[x_ == 0] = np.nan
        x_mean = np.nanmean(x_, axis=(0))
        x_median = np.nanmedian(x_, axis=(0))
        x_mean[np.isnan(x_mean)] = 0
        x_median[np.isnan(np.nan)] = 0

        X_mean = np.zeros(fc_imputed_x.shape)
        X_median = np.zeros(fc_imputed_x.shape)
        for k in range(X_mean.shape[0]):
          X_mean[k, :, :] = x_mean
          X_median[k, :, :] = x_median

        no, seq_len, dim = x.shape

        m_ = m.copy()
        # for k in range(no):
        #   for j in range(seq_len):
        #     if np.all(m[k, j, :] == 0):
        #       fc_imputed_x[k, j, :] = x_mean[j, :]
              # m_[k, j, :] = np.ones(x_mean[j, :].shape)

        #plot_heatmap_imstep(self.streams, fc_imputed_x, "4 fully connected output after point interpolation", train_test_label, self.run_id, self.missing_rate, self.seq_len, self.iteration, self.weather_str, self.n, i=i)

    return fc_imputed_x

  def fit(self, x, m, t, feature_labels):
    """Train the entire MRNN.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
    """
    # Train RNN part
    curves = []
    for f in range(self.dim):
      c = self.rnn_train(x, m, t, f)
      print('Finish ' + str(f+1) + f'-th out of {self.dim} features training with RNN for imputation')
      curves.append(c)

    plot_loss_curve(self.run_id, self.streams, self.n, self.start_i, self.end_i, curves, self.file, f"RNN for each feature training loss", self.seq_len,
                    self.missing_rate, self.iteration, self.learning_rate, self.weather_str, labels=feature_labels)

    plot_loss_curve(self.run_id, self.streams, self.n, self.start_i, self.end_i, [np.nanmean(curves, axis=0).tolist()], self.file, f"RNN mean training loss", self.seq_len,
                    self.missing_rate, self.iteration, self.learning_rate, self.weather_str)

    # Train FC part
    self.fc_train(x, m, t)
    print('Finish M-RNN training with both RNN and FC for imputation')

  def reshape_external_sample(self, X, meta, norm_parameters):
    x_test_list, m_test_list, t_test_list, stream_idx_list = [], [], [], []
    for i in range(X.shape[0]):
      x = X[i]
      mrnn_window = pd.read_csv(meta[i][3])
      n_stream_to_keep = self.x_train_shape[2]
      id = str(int(meta[i][0]))
      sample_stream = mrnn_window[id]
      mrnn_window = mrnn_window.drop(id, 1)
      mrnn_window = mrnn_window.iloc[:, :n_stream_to_keep-1]
      mrnn_window[id] = sample_stream

      assert np.array_equal(mrnn_window[id].values, x, equal_nan=True), 'error! could not find sample in mrnn window!'

      data_to_impute = mrnn_window.values
      stream_idx = mrnn_window.columns.tolist().index(id)
      stream_idx_list.append(stream_idx)

      #df = pd.concat([df] * (self.x_train_shape[2]), axis=1, ignore_index=True).values
      #data = MinMaxScaler_(df, norm_parameters)

      # Reverse time order
      data_to_impute = data_to_impute[::-1]

      no, dim = data_to_impute.shape
      # Define original data
      ori_x = list()
      #print("define original data...")
      for i in range(0, no, self.stride):
        start = i
        end = i + self.seq_len
        # print(start, end)
        temp_ori_x = data_to_impute[start: end]
        if temp_ori_x.shape[0] != self.seq_len:
          continue
        ori_x = ori_x + [temp_ori_x]
      # Introduce missingness
      m = list()
      x = list()
      t = list()
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
        temp_t = np.ones([self.seq_len, dim])
        for j in range(dim):
          for k in range(1, self.seq_len):
            if temp_m[k, j] == 0:
              temp_t[k, j] = temp_t[k - 1, j] + 1
        t = t + [temp_t]

      # Convert into 3d numpy array
      x = np.asarray(x)
      m = np.asarray(m)
      t = np.asarray(t)
      ori_x = np.asarray(ori_x)
      # Fill 0 to the missing values
      x = np.nan_to_num(x, 0)

      x_test = np.zeros(self.x_train_shape, dtype=np.float16)
      m_test = x_test.copy()
      t_test_ = initialise_time_matrix(np.empty(self.x_train_shape))
      t_test = t_test_.copy()

      x_test[0:x.shape[0], :, :] = x
      m_test[0:x.shape[0], :, :] = m
      t_test[0:x.shape[0], :, :] = t

      x_test_list.append(x)
      m_test_list.append(m)
      t_test_list.append(t)

    return x_test_list, m_test_list, t_test_list, stream_idx_list

  def transform_(self, X, meta, full_data, streams, days, norm_parameters):
    #print(X)
    print("applying mrnn to fold test data...")
    x_test, m_test, t_test, stream_idx_list = self.reshape_external_sample(X, meta, norm_parameters)
    imputed_list = []
    for i in tqdm(range(len(x_test))):
      imputed = self.transform(x_test[i], m_test[i], t_test[i], 0, tag="reshaped_testing")
      #imputed = Denormalization(imputed, norm_parameters)
      stream_idx = stream_idx_list[i]
      # s = str(int(meta[i][0]))
      # if s in streams:
      #   stream_idx = streams.index(s)
      # else:
      #   pass
        #print(f"mrnn did not train on stream {s}!")
      test_sample_before_imp = np.concatenate(x_test[i][0:days, :, 0]) #all streams are identical

      sample_imputed = np.concatenate(imputed[0:days, :, stream_idx])

      #resize sample to original size to offset mrnn edge effect
      size = X.shape[1]
      t = size - len(sample_imputed)
      sample_imputed = np.pad(sample_imputed, pad_width=(0, t), mode='constant')
      #inverse mrnn native preproc
      sample_imputed = sample_imputed[::-1]
      # plt.plot(sample_imputed)
      # plt.plot(test_sample_before_imp)
      # plt.show()

      imputed_list.append(sample_imputed)
    return np.array(imputed_list)

  def transform(self, x, m, t, i=0, tag=''):
    """Impute missing data using the entire MRNN.

    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix

    Returns:
      - imputed_x: imputed data
    """

    # print("making subplot...")
    # fig = make_subplots(
    #   rows=4,
    #   cols=x.shape[0],
    #   y_title="",
    #   x_title="Time (1 min bins)",
    # )
    #
    # for k in range(x.shape[0]):
    #   xx = x[k]
    #   mm = m[k]
    #   tt = t[k]
    #
    #   trace = go.Heatmap(
    #     z=xx.T,
    #     x=np.arange(0, xx.shape[0], 1),
    #     y=np.arange(0, xx.shape[0], 1),
    #     colorscale="Viridis",
    #     showscale=False
    #   )
    #   fig.append_trace(trace, row=1, col=k + 1)
    #
    #   trace = go.Heatmap(
    #     z=mm.T,
    #     x=np.arange(0, xx.shape[0], 1),
    #     y=np.arange(0, xx.shape[0], 1),
    #     colorscale="Viridis",
    #     showscale=False
    #   )
    #   fig.append_trace(trace, row=2, col=k + 1)
    #
    #   trace = go.Heatmap(
    #     z=tt.T,
    #     x=np.arange(0, xx.shape[0], 1),
    #     y=np.arange(0, xx.shape[0], 1),
    #     colorscale="Viridis",
    #     showscale=False
    #   )
    #   fig.append_trace(trace, row=3, col=k + 1)

    # Impute with both RNN and FC part
    imputed_x = self.rnn_fc_predict(x, m, t, "testing", i)

    # for k in range(imputed_x.shape[0]):
    #   trace = go.Heatmap(
    #     z=imputed_x[k].T,
    #     x=np.arange(0, xx.shape[0], 1),
    #     y=np.arange(0, xx.shape[0], 1),
    #     colorscale="Viridis",
    #     showscale=False
    #   )
    #   fig.append_trace(trace, row=4, col=k + 1)
    # out_dir = Path("testing") / f"{self.run_id}_missingrate_{self.missing_rate}_seql_{self.seq_len}_iteration_{self.iteration}_hw_{self.weather_str}_n_{self.n}"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # filename = f"{tag}_{i}_testing_block.html"
    # output = out_dir / filename
    # print(output)
    # if imputed_x.shape[0] == 1:
    #   fig.update_layout(
    #     autosize=True)
    # else:
    #   fig.update_layout(
    #     autosize=False,
    #     width=imputed_x.shape[0]*100,
    #     height=1080)
    # fig.write_html(str(output))
    #fig.write_image(str(output).replace(".html", ".png"))

    return imputed_x