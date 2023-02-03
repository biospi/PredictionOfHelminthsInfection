'''gain function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "gain: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
# import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from gain.helper import normalization, renormalization, rmse_loss, build_formated_axis
from gain.helper import xavier_init, restore_matrix_v1, restore_matrix_v2
from gain.helper import binary_sampler, uniform_sampler, sample_batch_index

import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from multiprocessing import Pool


def gain(xaxix_label, start_timestamp, miss_rate, out, thresh, ids, t_idx, output_dir, shape_o, rm_row_idx, data_m_x,
         imputed_data_x_li, data_x_o, data_x, gain_parameters, outpath, RESHAPE, ADD_TRANSP_COL, N_TRANSPOND, days, n_job):
    '''Impute missing values in data_x

  Args:
    - data_x: original data with missing values
    - gain_parameters: gain network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

  Returns:
    - imputed_data: imputed data
  '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

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

    ## gain architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## gain functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## gain structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## gain loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(1. - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## gain solver
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

    D_loss_list = []
    G_loss_list = []
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
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        D_loss_list.append(D_loss_curr)
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})
        G_loss_list.append(G_loss_curr)

        ## Return imputed data
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        M_mb = data_m
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)

        # Rounding
        # imputed_data = rounding(imputed_data, data_x)

        # if (i % 100 == 0) | (i == 0) | (i == range_iter[-1]):

        rmse_iter.append(i)
        i_d = imputed_data[:, :-N_TRANSPOND - 1 - 1]  # epoch and id
        fig = go.Figure(data=go.Heatmap(
            z=i_d,
            x=xaxix_label,
            y=np.array(list(range(i_d.shape[0]))),
            colorscale='Viridis'))
        fig.update_xaxes(tickformat="%H:%M")
        fig.update_layout(
            title="Activity data after imputation i=%s" % i,
            xaxis_title="Time (1 min bins)",
            yaxis_title="Transponders")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = output_dir + "/" + "imputed_gain.html"
        if i % 100 == 0:
            print(filename)
            fig.write_html(filename)

        df_ = pd.DataFrame(imputed_data)
        header = [str(x) for x in range(imputed_data.shape[1])]
        for v in range(1, N_TRANSPOND + 1):
            header[-v] = "t_%d" % (N_TRANSPOND - v)
        header[-N_TRANSPOND - 1] = "id"
        header[-N_TRANSPOND - 2] = "epoch"
        df_.columns = header
        dfs_transponder = [g for _, g in df_.groupby(['id'])]

        if it in [range_iter[0], range_iter[-1]]:#only export heatmaps for first and last iteartion
            pool = Pool(processes=n_job)
            for k in range(len(dfs_transponder)):
                pool.apply_async(worker_export_heatmap, (k, len(dfs_transponder), dfs_transponder[k],
                                                         N_TRANSPOND, start_timestamp, xaxix_label, thresh,
                                                         out, it))
            pool.close()
            pool.join()
            pool.terminate()


        # for k in range(len(dfs_transponder)):
        #   df_t_i = dfs_transponder[k].iloc[:, :-N_TRANSPOND - 2]
        #   valid = np.sum((~np.isnan(df_t_i.values)).astype(int))
        #   # if valid <= 0:
        #   #     continue
        #   id = int(dfs_transponder[k]["id"].values[0])
        #
        #   _, yaxis_label = build_formated_axis(start_timestamp, min_in_row=df_t_i.shape[1],
        #                                                  days_in_col=df_t_i.shape[0])
        #   fig = go.Figure(data=go.Heatmap(
        #     z=df_t_i.values,
        #     x=xaxix_label,
        #     #y=yaxis_label,
        #     y=np.arange(0, df_t_i.shape[1]),
        #     colorscale='Viridis'))
        #   fig.update_xaxes(tickformat="%H:%M")
        #   # fig.update_yaxes(tickformat="%d %b %Y")
        #   fig.update_layout(
        #     title="imputed %d thresh=%d iteration=%d" % (id, thresh, i),
        #     xaxis_title="Time (1 min bins)",
        #     yaxis_title="Samples")
        #   filename = out / f"{id}_imputed_reshaped_{thresh}_{k}_{valid}_iter_{i}.html"
        #   if i in [0, 99]:
        #       print(filename)
        #       fig.write_html(filename)

        '''

    RMSE Calculation

    '''

        if np.isnan(imputed_data).any():
            warnings.warn("Warning NaN in normalised imputed results.")

        if np.isnan(imputed_data).all():
            raise ValueError("Error while imputing data, all value NaN!")

        if RESHAPE:
            imputed_data_restored = restore_matrix_v1(i, thresh, xaxix_label, ids, start_timestamp, t_idx, out,
                                                      shape_o, rm_row_idx, imputed_data, N_TRANSPOND,
                                                      add_t_col=ADD_TRANSP_COL, days=days)
        else:
            imputed_data_restored = restore_matrix_v2(imputed_data, N_TRANSPOND)

        if miss_rate > 0:
            # rmse_g, rmse_l = rmse_loss(data_x_o.copy(), imputed_data.copy(), imputed_data_x_li.copy(), data_m_x, output_dir, i)
            rmse_g, rmse_l = rmse_loss(data_x, data_x_o.copy(), imputed_data_restored.copy(), imputed_data_x_li.copy(),
                                       data_m_x, output_dir, i)

            print('RMSE GAIN Performance: ' + str(np.round(rmse_g, 4)))
            print('RMSE LI Performance: ' + str(np.round(rmse_l, 4)))
            rmse_gain.append(rmse_g)
            rmse_li.append(rmse_l)

            rmse_info = {"rmse": [rmse_g], "rmse_li": [rmse_l], "training_shape": [data_x.shape], "i": [i],
                         "g_loss":[G_loss_curr], "d_loss": [D_loss_curr]}
            df_info = pd.DataFrame(rmse_info)
            filepath = outpath / f'rmse_{i}.csv'
            print(filepath)
            df_info.to_csv(filepath, index=False)
        i += 1
    epochs = list(range(iterations))
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    plt.plot(epochs, G_loss_list, label="generator loss")
    plt.plot(epochs, D_loss_list, label="discriminator loss")
    plt.legend()
    plt.title(f"GAN error function miss_rate={miss_rate:.2f}")
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    fig.savefig(outpath / 'gan_loss.png')

    if miss_rate > 0:
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        ax.set_ylabel('RMSE')
        ax.set_xlabel('iteration')
        plt.plot(rmse_iter, rmse_gain, label="RMSE GAIN", alpha=1)
        plt.plot(rmse_iter, rmse_li, label="RMSE LI", alpha=1)

        plt.title(f"RMSE iteration performance miss_rate={miss_rate:.2f}")
        plt.legend()
        filename = outpath / "RMSE.png"
        # print(filename)
        plt.savefig(filename)

    return imputed_data_restored, rmse_iter, rm_row_idx


def worker_export_heatmap(i, tot, transponder, N_TRANSPOND, start_timestamp, xaxix_label, thresh, out, it):
    df_t_i = transponder.iloc[:, :-N_TRANSPOND - 2]
    valid = np.sum((~np.isnan(df_t_i.values)).astype(int))
    # if valid <= 0:
    #     continue
    id = int(transponder["id"].values[0])
    print(f"exporting {id} imputed heatmaps {i}/{tot}...")
    _, yaxis_label = build_formated_axis(start_timestamp, min_in_row=df_t_i.shape[1],
                                         days_in_col=df_t_i.shape[0])
    fig = go.Figure(data=go.Heatmap(
        z=df_t_i.values,
        x=xaxix_label,
        # y=yaxis_label,
        y=np.arange(0, df_t_i.shape[1]),
        colorscale='Viridis'))
    fig.update_xaxes(tickformat="%H:%M")
    # fig.update_yaxes(tickformat="%d %b %Y")
    fig.update_layout(
        title="imputed %d thresh=%d iteration=%d" % (id, thresh, it),
        xaxis_title="Time (1 min bins)",
        yaxis_title="Samples")
    filename = out / f"{id}_imputed_reshaped_{thresh}_{i}_{valid}_iter_{it}.html"
    print(filename)
    fig.write_html(filename)