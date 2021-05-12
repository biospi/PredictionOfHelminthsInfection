import os
import sys

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
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def cnn2d(X_train_, X_test_, y_train_, y_test_ ):
    uci_har_signals_train, uci_har_labels_train, uci_har_signals_test, uci_har_labels_test = load_ucihar_data(X_train_, X_test_, y_train_, y_test_ )
    # print(uci_har_labels_test)

    scales = range(1, X_train_.shape[1])
    waveletname = 'morl'
    train_size = uci_har_signals_train.shape[0]
    test_size = uci_har_signals_test.shape[0]

    # train_size = 3
    # test_size = 3

    train_data_cwt = np.ndarray(shape=(train_size, uci_har_signals_train.shape[1]-1, uci_har_signals_train.shape[1], 1))

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
    test_data_cwt = np.ndarray(shape=(test_size, X_test_.shape[1]-1, X_test_.shape[1], 1))
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

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(10, 10)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
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
                  optimizer=keras.optimizers.Adam(),
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

    y = data_frame['label'].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[1:data_frame.shape[1] - 1]]
    test_size = 10
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, random_state=0, stratify=y, test_size=int(test_size)/100)
    cnn2d(X_train_, X_test_, y_train_, y_test_)


    folder_ucihar = 'F:/Data/CNN/UCI HAR Dataset/'
    # uci_har_signals_train, uci_har_labels_train, uci_har_signals_test, uci_har_labels_test = load_ucihar_data_(folder_ucihar)





