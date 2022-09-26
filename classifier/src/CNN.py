import os
import sys
import time
from multiprocessing import Manager, Pool
from sklearn.metrics import auc
import pywt
#from wavelets.wave_python.waveletFunctions import *
import itertools
import numpy as np
import pandas as pd
import sklearn
from scipy.fftpack import fft
import pycwt as wavelet
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import balanced_accuracy_score

from utils.Utils import plot_model_metrics
from utils.visualisation import plot_roc_range, plot_pr_range, plot_fold_details
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# def plot_wavelet(time, signal, scales,
#                  waveletname='cmor',
#                  cmap=plt.cm.seismic,
#                  title='Wavelet Transform (Power Spectrum) of signal',
#                  ylabel='Period (years)',
#                  xlabel='Time'):
#     plt.clf()
#     dt = time[1] - time[0]
#     [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
#     power = (abs(coefficients)) ** 2
#     period = 1. / frequencies
#     # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
#     # contourlevels = np.log2(levels)
#
#     fig, ax = plt.subplots(figsize=(15, 10))
#     im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both')
#
#     ax.set_title(title, fontsize=20)
#     ax.set_ylabel(ylabel, fontsize=18)
#     ax.set_xlabel(xlabel, fontsize=18)
#
#     yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
#     ax.set_yticks(np.log2(yticks))
#     ax.set_yticklabels(yticks)
#     ax.invert_yaxis()
#     ylim = ax.get_ylim()
#     ax.set_ylim(ylim[0], -1)
#
#     cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
#     fig.colorbar(im, cax=cbar_ax, orientation="vertical")
#     plt.show()

def plot_roc(name, labels, predictions, **kwargs):
  plt.clf()
  lab = np.argmax(labels, axis=1)
  pred = np.argmax(predictions, axis=1)
  fp, tp, _ = sklearn.metrics.roc_curve(lab, pred)
  plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  plt.legend(loc='lower right')
  plt.show()


def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(ax, time, signal, average_over = 5):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal + Time Average', fontsize=16)
    ax.legend(loc='upper right')

def plot_signal_plus_average(time, signal, average_over=5):
    fig, ax = plt.subplots(figsize=(15, 3))
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label='time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Signal Amplitude', fontsize=18)
    ax.set_title('Signal + Time Average', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    ax.legend()
    plt.show()


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def plot_fft_plus_power(time, signal):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1 / dt

    fig, ax = plt.subplots(figsize=(15, 3))
    variance = np.std(signal) ** 2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
    ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    ax.set_xlabel('Frequency [Hz / year]', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=18)
    ax.legend()
    plt.show()

def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels_ucihar(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


def load_ucihar_data_(folder):
    train_folder = folder + 'train/InertialSignals/'
    test_folder = folder + 'test/InertialSignals/'
    labelfile_train = folder + 'train/y_train.txt'
    labelfile_test = folder + 'test/y_test.txt'
    train_signals, test_signals = [], []
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels


def load_ucihar_data(X_train, X_test, y_train, y_test ):
    train_signals, test_signals = [], []
    # for input_file in os.listdir(train_folder):
    #     # signal = read_signals_ucihar(train_folder + input_file)

    train_signals.append(X_train.values.tolist())
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    # for input_file in os.listdir(test_folder):
    #     signal = read_signals_ucihar(test_folder + input_file)
    #     test_signals.append(signal)
    test_signals.append(X_test.values.tolist())
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = y_train.tolist()
    test_labels = y_test.tolist()
    return train_signals, train_labels, test_signals, test_labels


def load_matlab_dataset(fname, label_col='label'):
    print("load_df_from_datasets...", fname)
    data_frame = pd.read_csv(fname, sep=",", header=None, low_memory=False)
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    hearder[-1] = "label"
    data_frame.columns = hearder
    data_frame_original = data_frame.copy()
    data_frame = shuffle(data_frame)
    return data_frame_original, data_frame, data_frame


def get_norm_l2(data_frame_no_norm):
    """Apply l2 normalisation to each row in dataframe.

    Keyword arguments:
    data_frame_no_norm -- input raw dataframe containing samples (activity data, label/target)
    data_frame_mean -- mean dataframe containing median samples (mean activity data, label/target)
    """

    df_X_norm_l2 = pd.DataFrame(preprocessing.normalize(data_frame_no_norm.iloc[:, :-1]), columns=data_frame_no_norm.columns[:-1], dtype=float)
    df_X_norm_l2["label"] = data_frame_no_norm.iloc[:, -1]

    df_X_norm_l2_std = pd.DataFrame(preprocessing.StandardScaler(with_mean=True, with_std=False).fit_transform(df_X_norm_l2.iloc[:, :-1]), columns=data_frame_no_norm.columns[:-1], dtype=float)
    df_X_norm_l2_std["label"] = data_frame_no_norm.iloc[:, -1]

    return df_X_norm_l2


def load_df_from_datasets(fname, label_col='label'):
    print("load_df_from_datasets...", fname)
    # df = pd.read_csv(fname, nrows=1, sep=",", header=None, error_bad_lines=False)
    # # print(df)
    # type_dict = find_type_for_mem_opt(df)

    data_frame = pd.read_csv(fname, sep=",", header=None, low_memory=False)
    # print("shape before removal of duplicates=", data_frame.shape)
    # data_frame = data_frame.drop_duplicates()
    # print("shape after removal of duplicates=", data_frame.shape)
    # print(data_frame)
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]
    hearder[-19] = "label"
    hearder[-18] = "elem_in_row"
    hearder[-17] = "date1"
    hearder[-16] = "date2"
    hearder[-15] = "serial"
    hearder[-14] = "famacha_score"
    hearder[-13] = "previous_famacha_score"
    hearder[-12] = "previous_famacha_score2"
    hearder[-11] = "previous_famacha_score3"
    hearder[-10] = "previous_famacha_score4"

    hearder[-9] = "dtf1"
    hearder[-8] = "dtf2"
    hearder[-7] = "dtf3"
    hearder[-6] = "dtf4"
    hearder[-5] = "dtf5"

    hearder[-4] = "nd1"
    hearder[-3] = "nd2"
    hearder[-2] = "nd3"
    hearder[-1] = "nd4"

    data_frame.columns = hearder
    data_frame_original = data_frame.copy()
    cols_to_keep = hearder[:-19]
    cols_to_keep.append(label_col)
    data_frame = data_frame[cols_to_keep]
    data_frame = shuffle(data_frame)
    data_frame = data_frame.drop_duplicates()
    return data_frame_original, data_frame


def build_2dcnn_model(
    n_classes,
    input_shape
):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 5),
                     activation='relu', padding="same",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(10, 10)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def build_1dcnn_model(nb_classes, input_shape):
    padding = 'valid'
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
    conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

    conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
    conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

    flatten_layer = keras.layers.Flatten()(conv2)

    output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model


def cnn2d(X_train_, X_test_, y_train_, y_test_ ):
    uci_har_signals_train, uci_har_labels_train, uci_har_signals_test, uci_har_labels_test = load_ucihar_data(X_train_, X_test_, y_train_, y_test_ )
    # print(uci_har_labels_test)

    scales = range(1, 171)
    waveletname = 'morl'
    train_size = uci_har_signals_train.shape[0]
    test_size = uci_har_signals_test.shape[0]

    # train_size = 3
    # test_size = 3

    train_data_cwt = np.ndarray(shape=(train_size, len(scales), uci_har_signals_train.shape[1], 1))

    for ii in range(0, train_size):
        print(ii, train_size)
        jj = 0
        signal = uci_har_signals_train[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :X_train_.shape[1]]
        train_data_cwt[ii, :, :, jj] = coeff_

        # wavelet_type = 'morlet'
        # coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal, 1, wavelet=wavelet_type)
        # coefs_cc = np.conj(coefs)
        # power = np.real(np.multiply(coefs, coefs_cc))
        # power_flatten = power.flatten()
        #
        # plt.imshow(power, extent=[0, power.shape[1], 0, power.shape[0]], interpolation='nearest',
        #            aspect='auto')
        # plt.show()
    test_data_cwt = np.ndarray(shape=(test_size, len(scales), X_test_.shape[1], 1))
    for ii in range(0, test_size):
        print(ii, test_size)
        jj = 0
        signal = uci_har_signals_test[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :X_train_.shape[1]]
        test_data_cwt[ii, :, :, jj] = coeff_


    uci_har_labels_train = list(map(lambda x: int(x) - 1, uci_har_labels_train))
    uci_har_labels_test = list(map(lambda x: int(x) - 1, uci_har_labels_test))

    x_train = train_data_cwt
    y_train = list(uci_har_labels_train[:train_size])
    x_test = test_data_cwt
    y_test = list(uci_har_labels_test[:test_size])

    history = History()

    img_x = coeff_.shape[0]
    img_y = coeff_.shape[1]
    img_z = 1
    input_shape = (img_x, img_y, img_z)

    batch_size = 5
    num_classes = 2
    epochs = 10

    x_train = x_train.astype('float16')
    x_test = x_test.astype('float16')

    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 5),
                     activation='relu', padding="same",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(10, 10)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=METRICS)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

    # train_score = model.evaluate(x_train, y_train, verbose=0)
    # print('Train loss: {}, Train AUC: {}'.format(train_score[0], train_score[1]))
    # test_score = model.evaluate(x_test, y_test, verbose=0)

    baseline_results = model.evaluate(x_test, y_test,
                                      batch_size=5, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)

    test_predictions_baseline = model.predict(x_test, batch_size=5)
    plot_roc("Test Baseline", y_test, test_predictions_baseline, linestyle='--')


    # print('Test loss: {}, Test AUC: {}'.format(test_score[0], test_score[1]))
    # print("")
    #
    # y_pred = model.predict(x_test, batch_size=5, verbose=1)
    # y_pred_bool = np.argmax(y_pred, axis=1)
    # print("****************************************")
    # print("CNN                                     ")
    # y_test = np.argmax(y_test, axis=1)
    # print(classification_report(y_test, y_pred_bool))


def format_samples_for1dcnn(samples, y, num_classes):
    y = list(map(lambda x: int(x) - 1, y))
    data = np.ndarray(shape=(len(samples), len(samples[0]), 1), dtype=np.float16)
    for ii, s in enumerate(samples):
        # plt.imshow(
        #     matrix_2d,
        #     origin="lower",
        #     aspect="auto",
        #     interpolation="nearest",
        #     extent=[0, matrix_2d.shape[1], 0, matrix_2d.shape[0]]
        # )
        # plt.show()
        data[ii, :, 0] = s

    input_shape = (data.shape[1], 1)

    y = keras.utils.np_utils.to_categorical(y, num_classes)
    data = data.astype(np.float16)
    y = y.astype(np.float16)

    return data, y, input_shape


def format_samples_for2dcnn(samples, y, time_freq_shape, num_classes):
    y = list(map(lambda x: int(x) - 1, y))
    data = np.ndarray(shape=(len(samples), time_freq_shape[0], time_freq_shape[1], 1), dtype=np.float16)
    for ii, s in enumerate(samples):
        matrix_2d = s.reshape(time_freq_shape).astype(np.float16)
        # plt.imshow(
        #     matrix_2d,
        #     origin="lower",
        #     aspect="auto",
        #     interpolation="nearest",
        #     extent=[0, matrix_2d.shape[1], 0, matrix_2d.shape[0]]
        # )
        # plt.show()
        jj = 0
        data[ii, :, :, jj] = matrix_2d

    img_x = time_freq_shape[0]
    img_y = time_freq_shape[1]
    img_z = 1
    input_shape = (img_x, img_y, img_z)

    y = keras.utils.np_utils.to_categorical(y, num_classes)
    data = data.astype(np.float16)
    y = y.astype(np.float16)

    return data, y, input_shape


def fold_worker(
    info,
    out_dir,
    y_h,
    ids,
    meta,
    meta_data_short,
    sample_dates,
    days,
    steps,
    tprs_test,
    tprs_train,
    aucs_roc_test,
    aucs_roc_train,
    fold_results,
    fold_probas,
    label_series,
    mean_fpr_test,
    mean_fpr_train,
    clf_name,
    X,
    y,
    train_index,
    test_index,
    axis_test,
    axis_train,
    ifold,
    nfold,
    epochs,
    batch_size,
    time_freq_shape=None,
    cnnd=2
):
    print(f"process id={ifold}/{nfold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_h_train, y_h_test = y_h[train_index], y_h[test_index]
    ids_train, ids_test = ids[train_index], ids[test_index]

    meta_train, meta_test = meta[train_index], meta[test_index]
    meta_train_s, meta_test_s = (
        meta_data_short[train_index],
        meta_data_short[test_index],
    )

    sample_dates_train, sample_dates_test = (
        sample_dates[train_index],
        sample_dates[test_index],
    )
    class_healthy, class_unhealthy = 0, 1

    # hold all extra label
    fold_index = np.array(train_index.tolist() + test_index.tolist())
    X_fold = X[fold_index]
    y_fold = y[fold_index]
    ids_fold = ids[fold_index]
    sample_dates_fold = sample_dates[fold_index]
    meta_fold = meta[fold_index]

    # keep healthy and unhealthy only
    X_train = X_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    y_train = y_h_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    meta_train = meta_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    meta_train_s = meta_train_s[np.isin(y_h_train, [class_healthy, class_unhealthy])]

    X_test = X_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    y_test = y_h_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]

    meta_test = meta_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    meta_test_s = meta_test_s[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    ids_test = ids_test[np.isin(y_h_test, [class_healthy, class_unhealthy])]
    sample_dates_test = sample_dates_test[
        np.isin(y_h_test, [class_healthy, class_unhealthy])
    ]

    ids_train = ids_train[np.isin(y_h_train, [class_healthy, class_unhealthy])]
    sample_dates_train = sample_dates_train[
        np.isin(y_h_train, [class_healthy, class_unhealthy])
    ]

    start_time = time.time()
####################################################

    num_classes = len(np.unique(y_train))

    if cnnd == 1:
        x_train, y_train, input_shape = format_samples_for1dcnn(X_train, y_train, num_classes)
        x_test, y_test, input_shape = format_samples_for1dcnn(X_test, y_test, num_classes)
        model = build_1dcnn_model(
            num_classes, input_shape
        )

    if cnnd == 2:
        x_train, y_train, input_shape = format_samples_for2dcnn(X_train, y_train, time_freq_shape, num_classes)
        x_test, y_test, input_shape = format_samples_for2dcnn(X_test, y_test, time_freq_shape, num_classes)
        model = build_2dcnn_model(
            num_classes,
            input_shape
        )

    if os.name == 'nt': #plot_model requires to install graphviz on linux os but hpc wont let you use apt-get
        filepath = out_dir / f'cnn{cnnd}d_model.png'
        print(filepath)
        keras.utils.vis_utils.plot_model(
            model, to_file=filepath, show_shapes=False, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )

##############################################################

    # METRICS = [
    #     keras.metrics.TruePositives(name='tp'),
    #     keras.metrics.FalsePositives(name='fp'),
    #     keras.metrics.TrueNegatives(name='tn'),
    #     keras.metrics.FalseNegatives(name='fn'),
    #     keras.metrics.BinaryAccuracy(name='accuracy'),
    #     keras.metrics.Precision(name='precision'),
    #     keras.metrics.Recall(name='recall'),
    #     keras.metrics.AUC(name='auc'),
    # ]
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
    model.summary()

    model_dir = out_dir / f"cnn{cnnd}d_models"
    print(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_dir / f"best_model_{ifold}.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    fit_time = time.time() - start_time

    #todo clean up
    met = ""
    try:
        met = meta_test[0][7]
    except Exception as e:
        print(e)
    plot_model_metrics(history, out_dir, ifold, meta=met, dir_name=f"model_{cnnd}dcnn")

###############################################################
    model = keras.models.load_model(out_dir / f"cnn{cnnd}d_models" / f"best_model_{ifold}.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # test healthy/unhealthy
    y_pred = model.predict(x_test)
    y_pred_proba_test = y_pred
    y_pred = (y_pred[:, 1] >= 0.5).astype(int)

    y_test = y_test[:, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test[:, 1])
    axis_test.append({"fpr": fpr, "tpr": tpr})

    interp_tpr_test = np.interp(mean_fpr_test, fpr, tpr)
    interp_tpr_test[0] = 0.0
    tprs_test.append(interp_tpr_test)
    auc_value_test = auc(fpr, tpr)
    print("auc test=", auc_value_test)
    aucs_roc_test.append(auc_value_test)

    # viz_roc_train = plot_roc_curve(
    #     clf,
    #     X_train,
    #     y_train,
    #     label=None,
    #     alpha=0.3,
    #     lw=1,
    #     ax=None,
    #     c="tab:blue",
    # )
    # axis_train.append(viz_roc_train)
    x_train = x_train.astype(np.float16)
    y_pred_train = model.predict(x_train)
    y_pred_proba_train = y_pred_train
    y_pred_train = (y_pred_train[:, 1] >= 0.5).astype(int)

    y_train = y_train[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_pred_proba_train[:, 1])
    axis_train.append({"fpr": fpr, "tpr": tpr})

    interp_tpr_train = np.interp(mean_fpr_train, fpr, tpr)
    interp_tpr_train[0] = 0.0
    tprs_train.append(interp_tpr_train)
    auc_value_train = auc(fpr, tpr)
    print("auc train=", auc_value_train)
    aucs_roc_train.append(auc_value_train)

    # if ifold == 0:
    #     plot_high_dimension_db(
    #         out_dir / "testing",
    #         np.concatenate((X_train, X_test), axis=0),
    #         np.concatenate((y_train, y_test), axis=0),
    #         list(np.arange(len(X_train))),
    #         np.concatenate((meta_train_s, meta_test_s), axis=0),
    #         clf,
    #         days,
    #         steps,
    #         ifold,
    #     )
    #     plot_learning_curves(clf, X, y, ifold, out_dir / "testing")

    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    # print(f"y_test={y_test}")
    # print(f"y_pred={y_pred}")

    if np.array_equal(y_test, y_pred):
        print("perfect prediction!")
        #precision_recall_fscore_support returns same value when prediction is perfect
        precision = np.repeat(precision, 2)
        recall = np.repeat(recall, 2)
        fscore = np.repeat(fscore, 2)
        support = np.repeat(support, 2)

    print(f"precision={precision} recall={recall} fscore={fscore} support={support}")
    correct_predictions_test = (y_test == y_pred).astype(int)
    incorrect_predictions_test = (y_test != y_pred).astype(int)

    # data for training
    accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    (
        precision_train,
        recall_train,
        fscore_train,
        support_train,
    ) = precision_recall_fscore_support(y_train, y_pred_train)

    correct_predictions_train = (y_train == y_pred_train).astype(int)
    incorrect_predictions_train = (y_train != y_pred_train).astype(int)

    fold_result = {
        "i_fold": ifold,
        "info": info,
        "training_shape": X_train.shape,
        "testing_shape": X_test.shape,
        "target": int(class_unhealthy),
        "auc": auc_value_test,
        "accuracy": float(accuracy),
        "accuracy_train": float(accuracy_train),
        "class_healthy": int(class_healthy),
        "class_unhealthy": int(class_unhealthy),
        "y_test": y_test.tolist(),
        "y_pred_proba_test": y_pred_proba_test.tolist(),
        "y_pred_proba_train": y_pred_proba_train.tolist(),
        "ids_test": ids_test.tolist(),
        "ids_train": ids_train.tolist(),
        "sample_dates_test": sample_dates_test.tolist(),
        "sample_dates_train": sample_dates_train.tolist(),
        "meta_test": meta_test.tolist(),
        "meta_fold": meta_fold.tolist(),
        "meta_train": meta_train.tolist(),
        "correct_predictions_test": correct_predictions_test.tolist(),
        "incorrect_predictions_test": incorrect_predictions_test.tolist(),
        "correct_predictions_train": correct_predictions_train.tolist(),
        "incorrect_predictions_train": incorrect_predictions_train.tolist(),
        "test_precision_score_0": float(precision[0]),
        "test_precision_score_1": float(precision[1]),
        "test_recall_0": float(recall[0]),
        "test_recall_1": float(recall[1]),
        "test_fscore_0": float(fscore[0]),
        "test_fscore_1": float(fscore[1]),
        "test_support_0": float(support[0]),
        "test_support_1": float(support[1]),
        "train_precision_score_0": float(precision_train[0]),
        "train_precision_score_1": float(precision_train[1]),
        "train_recall_0": float(recall_train[0]),
        "train_recall_1": float(recall_train[1]),
        "train_fscore_0": float(fscore_train[0]),
        "train_fscore_1": float(fscore_train[1]),
        "train_support_0": float(support_train[0]),
        "train_support_1": float(support_train[1]),
        "fit_time": fit_time,
    }
    fold_results.append(fold_result)

    # test individual labels and store probabilities to be healthy/unhealthy
    print(f"process id={ifold}/{nfold} test individual labels...")
    for y_f in np.unique(y_fold):
        label = label_series[y_f]
        X_test = X_fold[y_fold == y_f]
        #x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype('float32')
        y_test = y_fold[y_fold == y_f]
        if cnnd == 1:
            x_test, y_test, input_shape = format_samples_for1dcnn(X_test, y_test, num_classes)
        if cnnd == 2:
            x_test, y_test, input_shape = format_samples_for2dcnn(X_test, y_test, time_freq_shape, num_classes)
        print(f"testing {label} X_test shape is {x_test.shape}...")
        y_pred_proba_test = model.predict(x_test)
        fold_proba = {
            "test_y_pred_proba_0": y_pred_proba_test[:, 0].tolist(),
            "test_y_pred_proba_1": y_pred_proba_test[:, 1].tolist(),
        }
        fold_probas[label].append(fold_proba)
    print(f"process id={ifold}/{nfold} done!")



def cross_validate_cnnnd(
    svc_kernel,
    out_dir,
    steps,
    cv_name,
    days,
    label_series,
    cross_validation_method,
    X,
    y,
    y_h,
    ids,
    meta,
    meta_columns,
    meta_data_short,
    sample_dates,
    clf_name,
    n_job,
    epochs=30,
    batch_size=32,
    time_freq_shape=None,
    cnnd=2,
    export_fig_as_pdf=False
):
    """Cross validate X,y data and plot roc curve with range
    Args:
        out_dir: output directory to save figures to
        steps: postprocessing steps
        cv_name: name of cross validation method
        days: count of activity days in sample
        label_series: dict that holds famacha label/target
        class_healthy: target integer of healthy class
        class_unhealthy: target integer of unhealthy class
        cross_validation_method: Cv object
        X: samples
        y: targets
    """
    scores, scores_proba = {}, {}
    plt.clf()
    fig_roc, ax_roc = plt.subplots(1, 2, figsize=(19.20, 6.20))
    fig_roc_merge, ax_roc_merge = plt.subplots(figsize=(12.80, 7.20))
    mean_fpr_test = np.linspace(0, 1, 100)
    mean_fpr_train = np.linspace(0, 1, 100)

    with Manager() as manager:
        # create result holders
        tprs_test = manager.list()
        tprs_train = manager.list()
        axis_test = manager.list()
        axis_train = manager.list()
        aucs_roc_test = manager.list()
        aucs_roc_train = manager.list()
        fold_results = manager.list()
        fold_probas = manager.dict()
        for k in label_series.values():
            fold_probas[k] = manager.list()

        #pool = Pool(processes=n_job)
        start = time.time()
        for ifold, (train_index, test_index) in enumerate(
            cross_validation_method.split(X, y)
        ):
            info = cross_validation_method.get_fold_info(ifold)
            fold_worker(
                info,
                out_dir,
                y_h,
                ids,
                meta,
                meta_data_short,
                sample_dates,
                days,
                steps,
                tprs_test,
                tprs_train,
                aucs_roc_test,
                aucs_roc_train,
                fold_results,
                fold_probas,
                label_series,
                mean_fpr_test,
                mean_fpr_train,
                clf_name,
                X,
                y,
                train_index,
                test_index,
                axis_test,
                axis_train,
                ifold,
                cross_validation_method.get_n_splits(),
                epochs,
                batch_size,
                time_freq_shape=time_freq_shape,
                cnnd=cnnd
            )
        #     pool.apply_async(
        #         fold_worker,
        #         (
        #             out_dir,
        #             y_h,
        #             ids,
        #             meta,
        #             meta_data_short,
        #             sample_dates,
        #             days,
        #             steps,
        #             tprs_test,
        #             tprs_train,
        #             aucs_roc_test,
        #             aucs_roc_train,
        #             fold_results,
        #             fold_probas,
        #             label_series,
        #             mean_fpr_test,
        #             mean_fpr_train,
        #             clf_name,
        #             X,
        #             y,
        #             train_index,
        #             test_index,
        #             axis_test,
        #             axis_train,
        #             ifold,
        #             cross_validation_method.get_n_splits(),
        #             epochs,
        #             batch_size
        #         ),
        #     )
        # pool.close()
        # pool.join()
        end = time.time()
        fold_results = list(fold_results)
        axis_test = list(axis_test)
        tprs_test = list(tprs_test)
        aucs_roc_test = list(aucs_roc_test)
        axis_train = list(axis_train)
        tprs_train = list(tprs_train)
        aucs_roc_train = list(aucs_roc_train)
        fold_probas = dict(fold_probas)
        fold_probas = dict([a, list(x)] for a, x in fold_probas.items())
        fit_test_time = "total time (s)= " + str(end - start)
        print(fit_test_time)
        time_file_path = out_dir / "fit_test_time.txt"
        print(time_file_path)
        with open(time_file_path, "w") as text_file:
            text_file.write(fit_test_time)

    plot_fold_details(fold_results, meta, meta_columns, out_dir)

    info = f"X shape:{str(X.shape)} healthy:{np.sum(y_h == 0)} unhealthy:{np.sum(y_h == 1)}"
    for a in axis_test:
        xdata = a["fpr"]
        ydata = a["tpr"]
        ax_roc[1].plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)
        ax_roc_merge.plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)

    for a in axis_train:
        xdata = a["fpr"]
        ydata = a["tpr"]
        ax_roc[0].plot(xdata, ydata, color="tab:blue", alpha=0.3, linewidth=1)
        ax_roc_merge.plot(xdata, ydata, color="tab:purple", alpha=0.3, linewidth=1)

    if cv_name == "LeaveOneOut":
        all_y = []
        all_probs = []
        for item in fold_results:
            all_y.extend(item['y_test'])
            all_probs.extend(np.array(item['y_pred_proba_test'])[:, 0])
        all_y = np.array(all_y)
        all_probs = np.array(all_probs)
        fpr, tpr, thresholds = roc_curve(all_y, all_probs)
        roc_auc = auc(fpr, tpr)
        ax_roc_merge.plot(fpr, tpr, lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % (roc_auc))
        ax_roc_merge.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
        ax_roc_merge.set_xlim([-0.05, 1.05])
        ax_roc_merge.set_ylim([-0.05, 1.05])
        ax_roc_merge.set_xlabel('False Positive Rate')
        ax_roc_merge.set_ylabel('True Positive Rate')
        ax_roc_merge.set_title('Receiver operating characteristic')
        ax_roc_merge.legend(loc="lower right")
        ax_roc_merge.grid()
        fig_roc.tight_layout()
        path = out_dir / "roc_curve" / cv_name
        path.mkdir(parents=True, exist_ok=True)
        tag = clf_name
        final_path = path / f"{tag}_roc_{steps}.png"
        print(final_path)
        fig_roc.savefig(final_path)

        final_path = path / f"{tag}_roc_{steps}_merge.png"
        print(final_path)
        fig_roc_merge.savefig(final_path)
    else:
        mean_auc = plot_roc_range(
            ax_roc_merge,
            ax_roc,
            tprs_test,
            mean_fpr_test,
            aucs_roc_test,
            tprs_train,
            mean_fpr_train,
            aucs_roc_train,
            out_dir,
            steps,
            fig_roc,
            fig_roc_merge,
            cv_name,
            days,
            info=info,
            tag=clf_name,
        )

    scores[f"{clf_name}_results"] = fold_results
    scores_proba[f"{clf_name}_probas"] = fold_probas

    print("export results to json...")
    filepath = out_dir / "results.json"
    print(filepath)
    with open(str(filepath), "w") as fp:
        json.dump(scores, fp)
    filepath = out_dir / "results_proba.json"
    print(filepath)
    with open(str(filepath), "w") as fp:
        json.dump(scores_proba, fp)

    return scores, scores_proba


if __name__ == "__main__":
    # dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
    # df_nino = pd.read_table(dataset)
    # N = df_nino.shape[0]
    # t0 = 1871
    # dt = 0.25
    # time = np.arange(0, N) * dt + t0
    # signal = df_nino.values.squeeze()
    #
    # scales = np.arange(1, 128)
    # plot_signal_plus_average(time, signal)
    # plot_fft_plus_power(time, signal)
    # plot_wavelet(time, signal, scales)

    file = "F:/Data/gen_dataset_debug/delmas_70101200027_1min_famachadays_1_threshold_interpol_30_threshold_zero2nan_480/training_sets/activity_delmas_70101200027_dbft_1_1min_threshi_30_threshz_480.csv"
    data_frame_original, data_frame = load_df_from_datasets(file)
    downsample_false_class = True
    if downsample_false_class:
        df_true = data_frame[data_frame['label'] == True]
        df_false = data_frame[data_frame['label'] == False]
        try:
            df_false = df_false.sample(df_true.shape[0])
        except ValueError as e:
            print(e)
            sys.exit(-1)
        data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)

    data_frame = get_norm_l2(data_frame)
    data_frame = data_frame.loc[[0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105], :]

    y = data_frame['label'].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[1:data_frame.shape[1] - 1]]



    test_size = 4
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, random_state=0, stratify=y, test_size=test_size)
    cnn2d(X_train_, X_test_, y_train_, y_test_)


    folder_ucihar = 'F:/Data/CNN/UCI HAR Dataset/'
    # uci_har_signals_train, uci_har_labels_train, uci_har_signals_test, uci_har_labels_test = load_ucihar_data_(folder_ucihar)





