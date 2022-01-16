"""Utility functions for MRNN modelling.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) process_batch_input_for_rnn: Convert tensor for rnn training
(2) initial_point_interpolation: Initial point interpolation
(3) BiGRUCell: Bidirectional GRU Cell
"""

import tensorflow as tf

import numpy as np
from tensorflow import keras
from tensorflow.python.keras.backend import placeholder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def process_batch_input_for_rnn(batch_input):
    """Convert tensor for rnn training.
  
  Args:
    - batch_input: original batch input
    
  Returns:
    - transformed_input: converted batch input for RNN
  """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    transformed_input = tf.transpose(batch_input_)
    return transformed_input


# def initial_point_interpolation(x, m, t, imputed_x, have_weather, fc=False):
#     """Initial point interpolation.
#
#     If the variable at time point 0 is missing, do zero-hold interpolation.
#
#     Args:
#         - x: original features
#         - m: masking matrix
#         - t: time information
#         - imputed_x: imputed data
#         - median_val: median values for imputation when nothing is observed
#
#     Returns:
#         - imputed_x: imputed and interpolated data
#     """
#
#     x_ = x.copy()
#     no, seq_len, dim = x.shape
#
#     x_[x_ == 0] = np.nan
#     x_mean = np.nanmean(x_, axis=(0))
#     x_median = np.nanmedian(x_, axis=(0))
#     x_mean[np.isnan(x_mean)] = 0
#     x_median[np.isnan(np.nan)] = 0
#
#     X_mean = np.zeros(imputed_x.shape)
#     X_median = np.zeros(imputed_x.shape)
#     for i in range(X_mean.shape[0]):
#         X_mean[i, :, :] = x_mean
#         X_median[i, :, :] = x_median
#
#     # for i in range(no):
#     #     for k in range(dim):
#     #         for j in range(seq_len):
#     #             # If there is no previous measurements
#     #             if t[i, j, k] > j:
#     #                 if np.max(m[i, :, k] == 1) == 1:
#     #                     idx = np.where(m[i, :, k] == 1)[0]
#     #                     # Do zero-hold interpolation
#     #                     imputed_x[i, j, k] = X_mean[i, j, k]
#     #                 # If nothing is measured
#     #                 else:
#     #                     imputed_x[i, j, k] = X_mean[i, j, k]
#
#     # for i in range(no):
#     #     for k in range(dim):
#     #         for j in range(seq_len):
#     #             if np.all(x[i, j, :] == 0):
#     #                 imputed_x[i, j, :] = X_mean[i, j, :]
#
#     c = 0
#     if "temperature" in have_weather:
#         c += 1
#     if "humidity" in have_weather:
#         c += 1
#
#     if c > 0:
#         for i in range(no):
#             for j in range(seq_len):
#                 if np.all(m[i, j, c:] == 0):
#                     imputed_x[i, j, c:] = x_mean[j, c:]
#     else:
#         for i in range(no):
#             for j in range(seq_len):
#                 if np.all(m[i, j, :] == 0):
#                     imputed_x[i, j, :] = x_mean[j, :]
#     return x

def initial_point_interpolation(x, m, t, imputed_x, wearther=None):
    """Initial point interpolation.

    If the variable at time point 0 is missing, do zero-hold interpolation.

    Args:
      - x: original features
      - m: masking matrix
      - t: time information
      - imputed_x: imputed data

    Returns:
      - imputed_x: imputed and interpolated data
    """

    no, seq_len, dim = x.shape

    for i in range(no):
        for k in range(dim):
            for j in range(seq_len):
                # If there is no previous measurements
                if (t[i, j, k] > j):
                    idx = np.where(m[i, :, k] == 1)[0]
                    # Do zero-hold interpolation
                    try:
                        imputed_x[i, j, k] = x[i, np.min(idx), k]
                    except Exception as e:
                        pass
                        #print(e)

    # scaler = StandardScaler()
    # imputed_x = scaler.fit_transform(imputed_x.reshape(-1, imputed_x.shape[-1])).reshape(imputed_x.shape)

    return imputed_x

# def initial_point_interpolation(x, m, t, imputed_x, weather):
#     """Initial point interpolation.
#
#   If the variable at time point 0 is missing, do zero-hold interpolation.
#
#   Args:
#     - x: original features
#     - m: masking matrix
#     - t: time information
#     - imputed_x: imputed data
#
#   Returns:
#     - imputed_x: imputed and interpolated data
#   """
#
#     no, seq_len, dim = x.shape
#     x_mean = np.mean(x, axis=(0))
#     X_mean = np.zeros(imputed_x.shape)
#     for i in range(X_mean.shape[0]):
#         X_mean[i, :, :] = x_mean
#
#     for i in range(no):
#         for k in range(dim):
#             for j in range(seq_len):
#                 # If there is no previous measurements
#                 if (t[i, j, k] > j):
#                     imputed_x[i, j, k] = X_mean[i, j, k]
#                     # idx = np.where(m[i, :, k] == 1)[0]
#                     # # Do zero-hold interpolation
#                     # try:
#                     #     imputed_x[i, j, k] = x[i, np.min(idx), k]
#                     # except Exception as e:
#                     #     #print(e)
#                     #     break
#
#     return imputed_x


class biGRUCell(object):
    """Bi-directional GRU cell object.
  
  Attributes:
    - input_size = Input Vector size
    - hidden_layer_size = Hidden layer size
    - target_size = Output vector size
  """

    def __init__(self, input_size, hidden_layer_size, target_size):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size

        # Weights and Bias for input and hidden tensor for forward pass
        self.Wr = tf.Variable(tf.zeros([self.input_size,
                                        self.hidden_layer_size]))
        self.Ur = tf.Variable(tf.zeros([self.hidden_layer_size,
                                        self.hidden_layer_size]))
        self.br = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wu = tf.Variable(tf.zeros([self.input_size,
                                        self.hidden_layer_size]))
        self.Uu = tf.Variable(tf.zeros([self.hidden_layer_size,
                                        self.hidden_layer_size]))
        self.bu = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wh = tf.Variable(tf.zeros([self.input_size,
                                        self.hidden_layer_size]))
        self.Uh = tf.Variable(tf.zeros([self.hidden_layer_size,
                                        self.hidden_layer_size]))
        self.bh = tf.Variable(tf.zeros([self.hidden_layer_size]))

        # Weights and Bias for input and hidden tensor for backward pass
        self.Wr1 = tf.Variable(tf.zeros([self.input_size,
                                         self.hidden_layer_size]))
        self.Ur1 = tf.Variable(tf.zeros([self.hidden_layer_size,
                                         self.hidden_layer_size]))
        self.br1 = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wu1 = tf.Variable(tf.zeros([self.input_size,
                                         self.hidden_layer_size]))
        self.Uu1 = tf.Variable(tf.zeros([self.hidden_layer_size,
                                         self.hidden_layer_size]))
        self.bu1 = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wh1 = tf.Variable(tf.zeros([self.input_size,
                                         self.hidden_layer_size]))
        self.Uh1 = tf.Variable(tf.zeros([self.hidden_layer_size,
                                         self.hidden_layer_size]))
        self.bh1 = tf.Variable(tf.zeros([self.hidden_layer_size]))

        # Weights for output layers
        self.Wo = tf.Variable(
            tf.random.truncated_normal(
                [self.hidden_layer_size * 2, self.target_size], mean=0.0, stddev=0.01, dtype=tf.dtypes.float32
            )
        )

        self.bo = tf.Variable(
            tf.random.truncated_normal(
                [self.target_size], mean=0.0, stddev=0.01, dtype=tf.dtypes.float32
            )
        )

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = placeholder(dtype=tf.float32,
                                   shape=[None, None, self.input_size],
                                   name='inputs')
        # Reversing the inputs by sequence for backward pass of the GRU
        self._inputs_rev = placeholder(dtype=tf.float32,
                                       shape=[None, None, self.input_size],
                                       name='inputs_rev')

        # Processing inputs to work with scan function
        self.processed_input = process_batch_input_for_rnn(self._inputs)
        # For bacward pass of the GRU
        self.processed_input_rev = process_batch_input_for_rnn(self._inputs_rev)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden,
                                        tf.zeros([input_size, hidden_layer_size]))

    def GRU_f(self, previous_hidden_state, x):
        """Function for Forward GRU cell.
    
    This function takes previous hidden state
    and memory tuple with input and
    outputs current hidden state.
    
    Args:
      - previous_hidden_state
      - x
      
    Returns:
      - current_hidden_state
    """
        # R Gate
        r = tf.sigmoid(tf.matmul(x, self.Wr) + \
                       tf.matmul(previous_hidden_state, self.Ur) + \
                       self.br)
        # U Gate
        u = tf.sigmoid(tf.matmul(x, self.Wu) + \
                       tf.matmul(previous_hidden_state, self.Uu) + \
                       self.bu)
        # Final Memory cell
        c = tf.tanh(tf.matmul(x, self.Wh) + \
                    tf.matmul(tf.multiply(r, previous_hidden_state), self.Uh) + \
                    self.bh)
        # Current Hidden state
        current_hidden_state = tf.multiply((1 - u), previous_hidden_state) + \
                               tf.multiply(u, c)

        return current_hidden_state

    def GRU_b(self, previous_hidden_state, x):
        """Function for Backward GRU cell.
    
    This function takes previous hidden
    state and memory tuple with input and
    outputs current hidden state.
    
    Args:
      - previous_hidden_state
      - x
      
    Returns:
      - current_hidden_state
    """
        # R Gate
        r = tf.sigmoid(tf.matmul(x, self.Wr1) + \
                       tf.matmul(previous_hidden_state, self.Ur1) + \
                       self.br1)
        # U Gate
        u = tf.sigmoid(tf.matmul(x, self.Wu1) + \
                       tf.matmul(previous_hidden_state, self.Uu1) + \
                       self.bu1)
        # Final Memory cell
        c = tf.tanh(tf.matmul(x, self.Wh1) + \
                    tf.matmul(tf.multiply(r, previous_hidden_state), self.Uh1) + \
                    self.bh1)
        # Current Hidden state
        current_hidden_state = tf.multiply((1 - u), previous_hidden_state) + \
                               tf.multiply(u, c)

        return current_hidden_state

    def get_states_f(self):
        """Function to get the hidden and memory cells after forward pass.
    
    Iterates through time/ sequence to get all hidden state
    
    Returns:
      - all_hidden_states
    """
        # Getting all hidden state through time
        all_hidden_states = tf.scan(self.GRU_f,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')
        return all_hidden_states

    def get_states_b(self):
        """Function to get the hidden and memory cells after backward pass.
    
    Iterates through time/ sequence to get all hidden state
    
    Returns:
      - all_hidden_states
    """
        all_hidden_memory_states = tf.scan(self.GRU_b,
                                           self.processed_input_rev,
                                           initializer=self.initial_hidden,
                                           name='states')
        # Now reversing the states to keep those in original order
        all_hidden_states = tf.reverse(all_hidden_memory_states, [1])
        return all_hidden_states

    def get_concat_hidden(self):
        """Function to concat the hiddenstates for backward and forward pass.
    
    Returns:
      - concat_hidden
    """
        # Getting hidden and memory for the forward pass
        all_hidden_states_f = self.get_states_f()
        # Getting hidden and memory for the backward pass
        all_hidden_states_b = self.get_states_b()
        # Concating the hidden states of forward and backward pass
        concat_hidden = tf.concat([all_hidden_states_f, all_hidden_states_b], 2)

        return concat_hidden

    def get_output(self, hidden_state):
        """Function to get output from a hidden layer.
    
    This function takes hidden state and returns output
    
    Returns:
      - output
    """
        output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self):
        """Function for getting all output layers.
    
    Iterating through hidden states to get outputs for all timestamp
    
    Returns:
      - all_outputs    
    """
        all_hidden_states = self.get_concat_hidden()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)

        return all_outputs
