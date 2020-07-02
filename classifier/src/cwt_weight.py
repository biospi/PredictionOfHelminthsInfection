import gc
import math
import os
import shutil
from sys import exit
from sklearn.multiclass import OneVsRestClassifier
import eli5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet
import pywt
from scipy.signal import chirp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions
import matplotlib.ticker as ticker
import pathlib
import json
import seaborn as sns
from scipy import signal
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import scikitplot as skplt
# sns.set()
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn import model_selection
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from classifier.src.my_lda import process_lda
from sklearn.linear_model import LassoCV, Lasso

DATA_ = []
CWT_RES = 1000000
TRAINING_DIR = "E:/Users/fo18103/PycharmProjects" \
               "/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/sd/"

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

plt.style.use("seaborn-white")
plt.rcParams["figure.facecolor"] = 'white'
plt.rcParams["savefig.facecolor"] = 'white'
plt.rcParams.update({'font.size': 12})


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        # i[i > 150] = -1
        s = pd.Series(i)
        s = s.interpolate(method='linear', limit_direction='both')
        # s = s.interpolate(method='spline', limit_direction='both')
        return s.tolist()
    except ValueError as e:
        print(e)
        return input_activity


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def dummy_sin():
    period = 5
    n = 1000
    t = np.linspace(0, period, n, endpoint=False)
    f0 = 1
    f1 = 10
    y = chirp(t, f0, period, f1, method='logarithmic')
    plt.plot(t, y)
    plt.grid(alpha=0.25)
    plt.xlabel('t (seconds)')
    plt.show()
    return t, y


def compute_cwt(activity, hd=True):
    if hd:
        return compute_cwt_hd(activity)
    else:
        return compute_cwt_sd(activity)


def compute_cwt_sd(activity):
    w = pywt.ContinuousWavelet('morl')
    scales = even_list(20)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    # indexes = np.asarray(list(range(coef.shape[1])))
    indexes = []
    return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt_hd(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    y = interpolate(y)

    delta_t = (x[1] - x[0]) * 1
    scales = np.arange(1, num_steps + 1) / 1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'
    # y = [0 if x is np.nan else x for x in y] #todo fix
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, wavelet=wavelet_type, freqs=freqs)

    # print("*********************************************")
    # print(y)
    # print(coefs)
    # print("*******************************************")
    # iwave = wavelet.icwt(coefs, scales, delta_t, wavelet=wavelet_type)
    # plt.plot(iwave)
    # plt.show()
    # plt.plot(activity)
    # plt.show()
    #
    # plt.matshow((coefs.real))
    # plt.show()
    # exit()
    cwt = [element for tupl in coefs.real for element in tupl]
    # indexes = np.asarray(list(range(len(coefs.real))))
    indexes = []
    return cwt, coefs.real, freqs, indexes, scales, delta_t, wavelet_type


META_DATA_LENGTH = 19


def find_type_for_mem_opt(df):
    data_col_n = df.iloc[[0]].size
    type_dict = {}
    for n, i in enumerate(range(0, data_col_n)):
        if n < (data_col_n - META_DATA_LENGTH):
            type_dict[str(i)] = np.float16
        else:
            type_dict[str(i)] = np.str
    del df
    type_dict[str(data_col_n - 1)] = np.int
    type_dict[str(data_col_n - 2)] = np.int
    type_dict[str(data_col_n - 3)] = np.int
    type_dict[str(data_col_n - 8)] = np.int
    type_dict[str(data_col_n - 9)] = np.int
    type_dict[str(data_col_n - 10)] = np.int
    type_dict[str(data_col_n - 11)] = np.int
    type_dict[str(data_col_n - 15)] = np.int
    return type_dict


def start(fname='', out_fname=None, fname_temp=None, fname_hum=None, resolution=None, days=None, f_config=None,
          farm_id=None, output_clf_transit=False, cwt1=None, cwt2=None, filter_delmas=False, filter_cedara=False,
          filter_resp_to_treat_delmas=False, filter_resp_to_treat_cedara=False):
    try:
        if fname_temp is not None:
            df_temp = pd.read_csv(fname_temp, sep=",", header=None)
            sample_count = df_temp.shape[1]
            hearder = [str(n) for n in range(0, sample_count)]
            df_temp.columns = hearder

        if fname_hum is not None:
            df_hum = pd.read_csv(fname_hum, sep=",", header=None)
            sample_count = df_hum.shape[1]
            hearder = [str(n) for n in range(0, sample_count)]
            df_hum.columns = hearder
    except FileNotFoundError as e:
        print("missing weather file!")
        print(e)

    print(out_fname)
    print("loading dataset...")
    print(fname)
    df = pd.read_csv(fname, nrows=1, sep=",", header=None)

    type_dict = find_type_for_mem_opt(df)
    df = pd.read_csv(fname, sep=",", header=None, dtype=type_dict)
    del type_dict

    # print(data_frame)
    sample_count = df.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
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

    df.columns = hearder
    print(df)

    if filter_resp_to_treat_delmas:
        print('filter_resp_to_treat_delmas...')
        df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd2 == 7)]
        filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score == 1))
        filter2 = ((df.famacha_score == 1) & (df.previous_famacha_score == 2) & (df.previous_famacha_score2 == 2))
        df.loc[filter1, 'label'] = 0
        df.loc[filter2, 'label'] = 1
        df1 = df[filter1]
        df2 = df[filter2]
        s = min([df1.shape[0], df2.shape[0]])
        df1 = df1.head(s)
        df2 = df2.head(s)
        df = pd.concat([df1, df2])

    if filter_resp_to_treat_cedara:
        print('filter_resp_to_treat_cedara...')
        df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd2 == 14)]
        filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score == 1))
        filter2 = ((df.famacha_score == 1) & (df.previous_famacha_score >= 2) & (df.previous_famacha_score2 >= 2))
        df.loc[filter1, 'label'] = 0
        df.loc[filter2, 'label'] = 1
        df1 = df[filter1]
        df2 = df[filter2]
        s = min([df1.shape[0], df2.shape[0]])
        df1 = df1.head(s)
        df2 = df2.head(s)
        df = pd.concat([df1, df2])


    if filter_delmas:
        print('filter_delmas...')
        if days == 7 + 7 + 7 + 7:
            # df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7) & (df.nd4 == 7)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score4 == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score3 == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2) & (df.previous_famacha_score2 == 2) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score4 == 1))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            df3 = df3.head(1)
            df = pd.concat([df1, df2, df3])

        if days == 7 + 7 + 7:
            df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2) & (df.previous_famacha_score2 == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            df3 = df3.head(1)
            df = pd.concat([df1, df2])

        if days == 7 + 7:
            df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1))
            filter3 = ((df.famacha_score == 1) & (df.previous_famacha_score == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            # df3 = df3.head(1)
            df = pd.concat([df1, df2])

        if days == 7:
            # df = df[(df.nd1 == 7) & (df.nd2 == 7)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1))
            filter3 = ((df.famacha_score == 1) & (df.previous_famacha_score == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            # df3 = df3.head(1)
            df = pd.concat([df1, df2])


    if filter_cedara:
        print('filter_delmas...')
        if days == 7 + 7 + 7 + 7:
            # df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7) & (df.nd4 == 7)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score4 == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score3 == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2) & (df.previous_famacha_score2 == 2) &
                       (df.previous_famacha_score3 == 1) & (df.previous_famacha_score4 == 1))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            df3 = df3.head(1)
            df = pd.concat([df1, df2, df3])

        if days == 7 + 7 + 7:
            df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd3 == 14)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2) & (df.previous_famacha_score2 == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            df3 = df3.head(1)
            df = pd.concat([df1, df2, df3])

        if days == 7 + 7:
            df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd3 == 14)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score >= 2) & (df.previous_famacha_score == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            # s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            df3 = df3.head(2)
            df = pd.concat([df1, df2, df3])

        if days == 7:
            # df = df[(df.nd1 == 14)]
            filter1 = ((df.famacha_score == 1) & (df.previous_famacha_score == 1))
            filter2 = ((df.famacha_score >= 2) & (df.previous_famacha_score == 1))
            filter3 = ((df.famacha_score == 2) & (df.previous_famacha_score == 2))
            df.loc[filter1, 'label'] = 0
            df.loc[filter2, 'label'] = 1
            df.loc[filter3, 'label'] = 2
            df1 = df[filter1]
            df2 = df[filter2]
            df3 = df[filter3]
            s = min([df1.shape[0], df2.shape[0]])
            # df1 = df1.head(s)
            # df2 = df2.head(s)
            # df3 = df3.head(2)
            df = pd.concat([df1, df2])

    print('labels')
    print(df['label'].value_counts())
    df.sort_index(inplace=False)
    print(df)
    df_0 = df

    df = df.loc[:, :'label']
    # df = shuffle(df)
    print(df)
    if df.shape[0] == 0:
        return

    # for n in [1,2,10,100,1000]:
    #     process2(df, df_0, out_fname, n_components=n, df_temp=df_temp, df_hum=df_hum, resolution=resolution, farm_id=farm_id, days=days,
    #             f_config=f_config, out_dir="%s\\%d\\" % (farm_id, days))
    #
    # exit()
    out_dir = "%s\\%d\\" % (farm_id, days)
    #
    # dfs_td, data = chunck_df(df, ignore=False)
    # dfs = []
    # for df_td in dfs_td:
    # df_cwt, _ = get_cwt_data_frame(df_td, df_0, out_fname, df_hum=df_hum, df_temp=df_temp, days=days, resolution=resolution, out_dir=out_dir)
    #     dfs.append(df_cwt)


    df_cwt, cwt_coefs_data = get_cwt_data_frame(df, df_0, out_fname, df_hum=df_hum, df_temp=df_temp, days=days, resolution=resolution,
                               out_dir=out_dir)
    dfs, data = chunck_df(df_cwt, cwt_coefs_data, ignore=True)


    explain_cwt(dfs, data, df, df_0, out_fname, df_temp=df_temp, df_hum=df_hum, resolution=resolution, farm_id=farm_id, days=days,
                f_config=f_config, out_dir=out_dir)

    print("output transit...")
    # dfs = []
    # for result in df_time_split:
    #     df, _, _, _, _, _, _, _, _, _ = result[0], result[1], result[2], result[3], result[4], result[5], result[6],\
    #                                     result[7], result[8], result[9]
    #     df = shuffle(df)
    #     dfs.append(df)
    # process_transit(dfs, days, resolution, farm_id)


def chunck_df(df, data, ignore=False):
    scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean = data

    X = df[df.columns[0:df.shape[1] - 1]]
    y = df['label']

    n_week = int(days/7)
    chunch_size = int((X.shape[1]/n_week)/1)
    W_DAY_STEP = 7
    step = int((X.shape[1]/(n_week*7))*W_DAY_STEP)

    print("step size is %d, chunch_size is %d, n_week is %d" % (step, chunch_size, n_week))
    dfs = []
    cwt_coefs_data = []
    if not ignore:

        for m, value in enumerate(range(0, int(X.shape[1]), step)):
            start = value
            end = int(start + chunch_size - 1)
            if end > int(X.shape[1]):
                end = int(X.shape[1]) - 1
            if abs(start - end) != chunch_size - 1:
                continue
            start = int(start)
            end = int(end)
            print("start=%d end=%d" % (start, end))
            df_x = pd.DataFrame(X.values[:, start:end])
            df_x['label'] = np.array(y)
            print("window:")
            print(df_x)
            dfs.append(df_x)

            # fig, axs = plt.subplots(2, 1, facecolor='white')
            # axs[0].pcolormesh(coefs_class0_mean[start:end], cmap='viridis')
            # axs[0].set_yscale('log')
            # axs[1].pcolormesh(coefs_class1_mean[start:end], cmap='viridis')
            # axs[1].set_yscale('log')
            # fig.show()
            # plt.close(fig)
            # plt.close()
            # fig.clear()

            cwt_coefs_data.append((scales, delta_t, wavelet_type, class0_mean[start:end],
                                   class1_mean[start:end], herd_mean[start:end], coefs_class0_mean[start:end],
                                   coefs_class1_mean[start:end], coefs_herd_mean[:, start:end]))
    else:
        df_x = pd.DataFrame(X.values[:, :])
        df_x['label'] = np.array(y)
        print("window:")
        print(df_x)
        dfs.append(df_x)
        cwt_coefs_data.append((scales, delta_t, wavelet_type, class0_mean,
                               class1_mean, herd_mean, coefs_class0_mean,
                               coefs_class1_mean, coefs_herd_mean[:, :]))

    return dfs, cwt_coefs_data


def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    # if output_dim not in [1, 2, 3]:
    #     raise ValueError("available dimension for features reduction are 1, 2 and 3.")
    if output_dim == 3:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])))
        y_train = np.append(y_train, (3, 4))
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1]), np.ones(X_train.shape[1])])))
        y_test = np.append(y_test, (3, 4))
    if output_dim == 2:
        X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
        y_train = np.append(y_train, 3)
        X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
        y_test = np.append(y_test, 3)
    clf = LDA(n_components=output_dim)
    X_train = clf.fit_transform(X_train, y_train)
    X_test = clf.fit_transform(X_test, y_test)
    if output_dim != 1:
        X_train = X_train[0:-(output_dim - 1)]
        y_train = y_train[0:-(output_dim - 1)]
        X_test = X_test[0:-(output_dim - 1)]
        y_test = y_test[0:-(output_dim - 1)]

    return X_train, X_test, y_train, y_test, clf


def process_fold(n, X, y, i, dim_reduc=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i, stratify=y)
    print(X_train.shape, X_test.shape, y)

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == 'LDA':
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_lda(n, X_train, X_test, y_train,
                                                                                      y_test)

    print(X_train_reduced.shape, X_test_reduced.shape, y)
    X_reduced = np.concatenate((X_train_reduced, X_test_reduced), axis=0)
    print(y_train_reduced.shape, y_test_reduced.shape)
    y_reduced = np.concatenate((y_train_reduced, y_test_reduced), axis=0)

    return X_reduced, y_reduced, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced


def get_proba(y_probas, y_pred):
    class_0 = []
    class_1 = []
    for i, item in enumerate(y_probas):
        if y_pred[i] == 0:
            class_0.append(item[0])
        if y_pred[i] == 1:
            class_1.append(item[1])

    class_0 = np.asarray(class_0)
    class_1 = np.asarray(class_1)

    return np.mean(class_0), np.mean(class_1)


def get_prec_recall_fscore_support(test_y, pred_y):
    precision_recall_fscore_support_result = precision_recall_fscore_support(test_y, pred_y, average=None,
                                                                             labels=[0, 1])
    precision_false = precision_recall_fscore_support_result[0][0]
    precision_true = precision_recall_fscore_support_result[0][1]
    recall_false = precision_recall_fscore_support_result[1][0]
    recall_true = precision_recall_fscore_support_result[1][1]
    fscore_false = precision_recall_fscore_support_result[2][0]
    fscore_true = precision_recall_fscore_support_result[2][1]
    support_false = precision_recall_fscore_support_result[3][0]
    support_true = precision_recall_fscore_support_result[3][1]
    return precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true


def plot_2D_decision_boundaries(X_lda, y_lda, X_test, y_test, title, clf, filename="", days=None, resolution=None,
                                folder=None, i=0, df_id=0, sub_dir_name=None, n_bin=8, save=True):
    print('graph...')
    # plt.subplots_adjust(top=0.75)
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    fig, ax = plt.subplots(figsize=(7., 4.8))
    # plt.subplots_adjust(top=0.75)
    min = abs(X_lda.min()) + 1
    max = abs(X_lda.max()) + 1
    print(X_lda.shape)
    print(min, max)
    if np.max([min, max]) > 100:
        return
    xx, yy = np.mgrid[-min:max:.01, -min:max:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    offset_r = 0
    offset_g = 0
    offset_b = 0
    colors = [((77+offset_r)/255, (157+offset_g)/255, (210+offset_b)/255),
              (1, 1, 1),
              ((255+offset_r)/255, (177+offset_g)/255, (106+offset_b)/255)]
    cm = LinearSegmentedColormap.from_list('name', colors, N=n_bin)

    for _ in range(0, 1):
        contour = ax.contourf(xx, yy, probs, n_bin, cmap=cm, antialiased=False, vmin=0, vmax=1, alpha=0.3, linewidth=0,
                              linestyles='dashed', zorder=-1)
        ax.contour(contour, cmap=cm, linewidth=1, linestyles='dashed', zorder=-1, alpha=1)

    ax_c = fig.colorbar(contour)

    ax_c.set_alpha(1)
    ax_c.draw_all()

    ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, 0.5, 0.75, 1])
    # ax_c.ax.set_yticklabels(['0', '0.15', '0.3', '0.45', '0.6', '0.75', '0.9', '1'])

    X_lda_0 = X_lda[y_lda == 0]
    X_lda_1 = X_lda[y_lda == 1]

    X_lda_0_t = X_test[y_test == 0]
    X_lda_1_t = X_test[y_test == 1]
    marker_size = 150
    ax.scatter(X_lda_0[:, 0], X_lda_0[:, 1], c=(39/255, 111/255, 158/255), s=marker_size, vmin=-.2, vmax=1.2,
               edgecolor=(49/255, 121/255, 168/255), linewidth=0, marker='s', alpha=0.7, label='Class0 (Healthy)'
               , zorder=1)

    ax.scatter(X_lda_1[:, 0], X_lda_1[:, 1], c=(251/255, 119/255, 0/255), s=marker_size, vmin=-.2, vmax=1.2,
               edgecolor=(255/255, 129/255, 10/255), linewidth=0, marker='^', alpha=0.7, label='Class1 (Unhealthy)'
               , zorder=1)

    ax.scatter(X_lda_0_t[:, 0], X_lda_0_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', label='Test data', zorder=1)

    ax.scatter(X_lda_1_t[:, 0], X_lda_1_t[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', zorder=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6, linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    handles, labels = ax.get_legend_handles_labels()
    db_line = Line2D([0], [0], color=(183/255, 37/255, 42/255), label='Decision boundary')
    handles.append(db_line)

    plt.legend(loc=4, fancybox=True, framealpha=0.4, handles=handles)
    plt.title(title)
    ttl = ax.title
    ttl.set_position([.57, 0.97])
    # plt.tight_layout()

    # path = filename + '\\' + str(resolution) + '\\'
    # path_file = path + "%d_p.png" % days
    # pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(path_file, bbox_inches='tight')
    if save:
        path = "%s/%s/decision_boundaries_graphs/df%d/" % (folder, sub_dir_name, df_id)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "iter_%d.png" % (i)
        final_path = '%s/%s' % (path, filename)
        print(final_path)
        try:
            plt.savefig(final_path, bbox_inches='tight')
        except FileNotFoundError as e:
            print(e)
            exit()

        plt.close()
        # fig.show()
        plt.close()
        fig.clear()
    else:
        fig.show()


def plot_2D_decision_boundaries_(X, y, X_test, title, clf, folder=None, i=0, df_id=None, sub_dir_name=None, save=True):

    # plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    #
    # # plot the decision function
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    #
    # # create grid to evaluate model
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = clf.decision_function(xy).reshape(XX.shape)
    #
    # # plot decision boundary and margins
    # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'])
    # # plot support vectors
    # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k')
    # plt.show()




    fig = plt.figure(figsize=(8, 7), dpi=100)
    plt.subplots_adjust(top=0.80)
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}
    plot_decision_regions(X, y, clf=clf, legend=2,
                          X_highlight=X_test,
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    if save:
        path = "%s/%s/decision_boundaries_graphs/df%d/" % (folder, sub_dir_name, df_id)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "iter_%d.png" % (i)
        final_path = '%s/%s' % (path, filename)
        print(final_path)
        try:
            plt.savefig(final_path)
        except FileNotFoundError as e:
            print(e)
            exit()

        plt.close()
        # fig.show()
        return final_path
    else:
        plt.show()


def compute_model_loo(X, y, X_train, y_train, X_test, y_test, farm_id, n, clf=None, dim_reduc_name="LDA", resolution="10min", df_id=None, days=None):
    # X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(2, X, y, n, dim_reduc=dim_reduc_name)
    print("fitting...")
    clf.fit(X_train, y_train)
    # clf = clf.best_estimator_
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, \
    support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if np.isnan(recall_false):
        recall_false = -1
    if np.isnan(recall_true):
        recall_true = -1
    if np.isnan(p_y_false):
        p_y_false = -1
    if np.isnan(p_y_true):
        p_y_true = -1

    print(('LREG', '' if dim_reduc_name is None else dim_reduc_name, 2, 3, 0,
           acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
           p_y_false * 100, p_y_true * 100,
           np.count_nonzero(y == 0), np.count_nonzero(y == 1),
           np.count_nonzero(y == 0), np.count_nonzero(y == 1),
           np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
           resolution))

    title = '%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s\n' % (
                'LREG', '' if dim_reduc_name is None else dim_reduc_name, 2, 3, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false * 100, p_y_true * 100,
                np.count_nonzero(y == 0), np.count_nonzero(y == 1),
                np.count_nonzero(y == 0), np.count_nonzero(y == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                resolution)

    sub_dir_name = "days_%d_class0_%d_class1_%d" % (days, np.count_nonzero(y == 0), np.count_nonzero(y == 1))

    plot_2D_decision_boundaries(X, y, X_test, y_test, title, clf,
                                folder='%s\\%d\\transition\\classifier_transit' % (farm_id, days), i=n, df_id=df_id,
                                sub_dir_name=sub_dir_name)
    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name



def compute_model(X, y, n, farm_id, clf=None, dim_reduc_name="LDA", resolution="10min", df_id=None, days=None):
    # X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(2, X, y, n, dim_reduc=dim_reduc_name)
    print("fitting...")
    X_test = X
    y_test = y
    # clf.fit(X_train, y_train)
    # clf = clf.best_estimator_
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, \
    support_false, support_true = get_prec_recall_fscore_support(
        y_test, y_pred)

    if np.isnan(recall_false):
        recall_false = -1
    if np.isnan(recall_true):
        recall_true = -1
    if np.isnan(p_y_false):
        p_y_false = -1
    if np.isnan(p_y_true):
        p_y_true = -1

    print(('LREG', '' if dim_reduc_name is None else dim_reduc_name, 2, 3, 0,
           acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
           p_y_false * 100, p_y_true * 100,
           np.count_nonzero(y == 0), np.count_nonzero(y == 1),
           np.count_nonzero(y == 0), np.count_nonzero(y == 1),
           np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
           resolution))

    title = '%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s\n' % (
                'LREG', '' if dim_reduc_name is None else dim_reduc_name, 2, 3, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false * 100, p_y_true * 100,
                np.count_nonzero(y == 0), np.count_nonzero(y == 1),
                np.count_nonzero(y == 0), np.count_nonzero(y == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                resolution)

    sub_dir_name = "days_%d_class0_%d_class1_%d" % (days, np.count_nonzero(y == 0), np.count_nonzero(y == 1))

    plot_2D_decision_boundaries(X, y, X_test, y_test, title, clf,
                                folder='%s\\%d\\transition\\classifier_transit' % (farm_id, days), i=n, df_id=df_id,
                                sub_dir_name=sub_dir_name)

    print(n, acc)
    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name


def get_conf_interval(tprs, mean_fpr):
    confidence_lower = []
    confidence_upper = []
    df_tprs = pd.DataFrame(tprs, dtype=float)
    for column in df_tprs:
        scores = df_tprs[column].values.tolist()
        scores.sort()
        upper = np.percentile(scores, 95)
        confidence_upper.append(upper)
        lower = np.percentile(scores, 0.025)
        confidence_lower.append(lower)

    confidence_lower = np.asarray(confidence_lower)
    confidence_upper = np.asarray(confidence_upper)
    # confidence_upper = np.minimum(mean_tpr + std_tpr, 1)
    # confidence_lower = np.maximum(mean_tpr - std_tpr, 0)

    return confidence_lower, confidence_upper


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, i, fig):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    ax.fill_between(mean_fpr, confidence_lower, confidence_upper, color='tab:blue', alpha=.2)
                    #label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           # title="Receiver operating characteristic iteration %d" % (i + 1)
           )
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/" % (out_dir)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'roc_%d.png' % i)
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    fig.savefig(final_path)


def process_transit(dfs, days, resolution, farm_id):
    data_acc, data_pf, data_pt, data_rf, data_rt, data_ff, data_ft, data_sf, data_st = {}, {}, {}, {}, {}, {}, {}, {}, {}
    sub_dir_name = None
    for id, data_frame in enumerate(dfs):
        # kf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
        # param_grid = {'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # clf = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), param_grid)
        # clf = LogisticRegression(n_jobs=8)

        # N_ITER = 1000
        # model = BaggingRegressor(LogisticRegression(),
        #                          n_estimators=N_ITER,
        #                          bootstrap=True, n_jobs=8)

        # clf = SVC(kernel='linear', probability=True)
        X, y = process_data_frame_(data_frame)
        X, _, y, _, clf = reduce_lda(2, X, X, y, y)
        # model.fit(X, y)
        acc_list, p_f_list, p_t_list, recall_f_list, recall_t_list, fscore_f_list, fscore_t_list, support_f_list, support_t_list = [], [], [], [], [], [], [], [], []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        # loo = LeaveOneOut()
        # loo.get_n_splits(X, y)
        rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=int((datetime.now().microsecond) / 10))


        for n, (train_index, test_index) in enumerate(rkf.split(X)):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name = compute_model_loo(
                X, y, X_train, y_train, X_test, y_test, farm_id, n, clf=clf, df_id=id, days=days)

        # for n, clf in enumerate(model.estimators_):
        #     try:
        #         acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name = compute_model(
        #             X, y, n, farm_id, clf=clf, df_id=id, days=days)
        #     except ValueError as e:
        #         print(e)
        #         continue

            viz = plot_roc_curve(clf, X, y,
                                 name='',
                                 label='_Hidden',
                                 alpha=0, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            acc_list.append(acc)
            p_f_list.append(precision_false)
            p_t_list.append(precision_true)
            recall_f_list.append(recall_false)
            recall_t_list.append(recall_true)
            fscore_f_list.append(fscore_false)
            fscore_t_list.append(fscore_true)
            support_f_list.append(support_false)
            support_t_list.append(support_true)
        out_dir = "%s\\%d\\" % (farm_id, days)
        plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, id, fig)
        fig.clear()
        print("acc_list", np.mean(acc_list), acc_list)
        data_acc[id] = acc_list
        data_pf[id] = p_f_list
        data_pt[id] = p_t_list
        data_rf[id] = recall_f_list
        data_rt[id] = recall_t_list
        data_ff[id] = fscore_f_list
        data_ft[id] = fscore_t_list
        data_sf[id] = support_f_list
        data_st[id] = support_t_list



    ribbon_plot_dir = '%s\\%d\\transition\\ribbon_transit\\%s' % (farm_id, days, sub_dir_name)
    pathlib.Path(ribbon_plot_dir).mkdir(parents=True, exist_ok=True)

    plot_(ribbon_plot_dir, data_acc, 'Classifier accuracy over time during increase of the FAMACHA score',
          "model accuracy in %", days)
    # plot_(ribbon_plot_dir, data_pf, 'Classifier precision(False) over time during increase of the FAMACHA score',
    #       "model precision(False) in %")
    # plot_(ribbon_plot_dir, data_pt, 'Classifier precision(True) over time during increase of the FAMACHA score',
    #       "model precision(True) in %")
    # plot_(ribbon_plot_dir, data_rf, 'Classifier recall(False) over time during increase of the FAMACHA score',
    #       "model recall(False) in %")
    # plot_(ribbon_plot_dir, data_rt, 'Classifier recall(True) over time during increase of the FAMACHA score',
    #       "model recall(True) in %")
    # plot_(ribbon_plot_dir, data_ff, 'Classifier Fscore(False) over time during increase of the FAMACHA score',
    #       "model Fscore(False) in %")
    # plot_(ribbon_plot_dir, data_ft, 'Classifier Fscore(True) over time during increase of the FAMACHA score',
    #       "model Fscore(True) in %")
    # plot_(ribbon_plot_dir, data_sf, 'Classifier Support(False) over time during increase of the FAMACHA score',
    #       "model Support(False)")
    # plot_(ribbon_plot_dir, data_st, 'Classifier Support(True) over time during increase of the FAMACHA score',
    #       "model Support(True)")

    # fig = plt.figure(figsize=(10, 25))
    # ax = fig.add_subplot(711)
    # plot(ax, data_acc, 'Classifier accuracy over time during increase of the FAMACHA score', "model accuracy in %")
    # ax = fig.add_subplot(712)
    # plot(ax, data_pf, 'Classifier precision(False) over time during increase of the FAMACHA score',
    #      "model precision(False) in %")
    # ax = fig.add_subplot(713)
    # plot(ax, data_pt, 'Classifier precision(True) over time during increase of the FAMACHA score',
    #      "model precision(True) in %")
    # ax = fig.add_subplot(714)
    # plot(ax, data_rf, 'Classifier recall(False) over time during increase of the FAMACHA score',
    #      "model recall(False) in %")
    # ax = fig.add_subplot(715)
    # plot(ax, data_rt, 'Classifier recall(True) over time during increase of the FAMACHA score',
    #      "model recall(True) in %")
    # ax = fig.add_subplot(716)
    # plot(ax, data_ff, 'Classifier Fscore(False) over time during increase of the FAMACHA score',
    #      "model Fscore(False) in %")
    # ax = fig.add_subplot(717)
    # plot(ax, data_ft, 'Classifier Fscore(True) over time during increase of the FAMACHA score',
    #      "model Fscore(True) in %")
    # ax = fig.add_subplot(918)
    # plot(ax, data_sf, 'Classifier Support(False) over time during increase of the FAMACHA score', "model Support(False)")
    # ax = fig.add_subplot(919)
    # plot(ax, data_st, 'Classifier Support(True) over time during increase of the FAMACHA score', "model Support(True)")
    # plt.tight_layout()
    # plt.show()
    # path = "%s\\model_transit.png" % ribbon_plot_dir
    # print(path)
    # fig.savefig(path, dpi=100)
    # fig.clear()
    # plt.close(fig)


def plot_(path, data, title, y_label, days):
    df = pd.DataFrame.from_dict(data, orient='index')
    print(df)
    time = []
    acc = []
    for index, row in df.iterrows():
        print(row[0], row[1])
        for n in range(df.shape[1]):
            time.append(index)
            acc.append(row[n])
    data_dict = {'time': time, 'acc': acc}
    df = pd.DataFrame.from_dict(data_dict)
    print(df)
    time_axis = interpolate_time(np.arange(days + 1), len(df["time"]))
    time_axis = time_axis.tolist()
    time_axis_s = []
    for t in time_axis:
        time_axis_s.append("%d" % t)
    ax = sns.lineplot(x=df["time"], y="acc", data=df, marker="o")
    ax.set_title(title)
    # ax = df.copy().plot.box(grid=True, patch_artist=True, title=title, figsize=(10, 7))
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels_ = interpolate_time(np.arange(days + 1), len(labels))
    for i, item in enumerate(labels_):
        labels[i] = "%d" % math.ceil(float(item))

    ax.set_xticklabels(labels)

    # ax.set_xticklabels(time_axis_s)
    file_path = '%s\\%s.png' % (path, y_label)
    plt.savefig(file_path)
    plt.show()


def plot(ax, data, title, y_label):
    df = pd.DataFrame.from_dict(data, orient='index')
    print(df)
    time = []
    acc = []
    for index, row in df.iterrows():
        print(row[0], row[1])
        for n in range(df.shape[1]):
            time.append(index)
            acc.append(row[n])
    data_dict = {'time': time, 'acc': acc}
    df = pd.DataFrame.from_dict(data_dict)
    print(df)
    ax = sns.lineplot(x="time", y="acc", data=df)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    # df.plot.box(grid=True, patch_artist=True, title=title, ax=ax, stacked=True)


def mean(a):
    return sum(a) / len(a)


def interpolate_(array):
    nans, x = nan_helper(array)
    array[nans] = np.interp(x(nans), x(~nans), array[~nans])
    return array


def to_list_of_nparray(l):
    result = []
    for array in l:
        array = np.asarray(array, dtype=np.float16)
        array = interpolate(array)
        result.append(array)

    return result


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def contains_negative(list):
    for v in list:
        if v is None:
            continue
        if v < 0:
            print("negative number found!", v)
            exit()
            return True
    return False


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except Exception:
        print("file not found.")


def process_data_frame_(data_frame, y_col='label'):
    data_frame = data_frame.fillna(-1)
    # cwt_shape = data_frame[data_frame.columns[0:2]].values
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - 1]].values
    print(X)
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    return X, y


def get_cwt_data_frame(data_frame, data_frame_0, out_fname=None, df_hum=None, df_temp=None, resolution=None,
                       days=None, out_dir=None):
    global DATA_
    DATA_ = []
    print(out_fname)
    # results = []
    #data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    # X_t = df_temp[df_temp.columns[0:df_temp.shape[1] - 1]].values
    # X_h = df_hum[df_hum.columns[0:df_hum.shape[1] - 1]].values
    # X_date = data_frame_0[data_frame_0.columns[data_frame_0.shape[1] - 9:data_frame_0.shape[1]-4]].values
    # cwt_list = []
    # data_frame_0 = data_frame_0.reset_index(drop=True)

    H = []
    for _, activity in enumerate(X):
        activity = interpolate(activity)
        activity = np.asarray(activity)
        H.append(activity)
    herd_mean = np.average(H, axis=0)
    print("herd window:")
    print(pd.DataFrame(herd_mean).transpose())

    print("finished computing herd mean.")

    print("computing herd cwt")
    cwt_herd, coefs_herd_mean, freqs_h, _, _, _, _ = compute_cwt(herd_mean)
    DATA_.append({'coef_shape': coefs_herd_mean.shape, 'freqs': freqs_h})
    print("finished calculating herd cwt.")

    purge_file(out_fname)
    X_cwt = pd.DataFrame()
    cpt = 0
    # with open(out_fname, 'a') as outfile:
    class0 = []
    class1 = []
    class0_t = []
    class1_t = []
    for activity, (i, row) in zip(X, data_frame.iterrows()):
        meta = row["label":].values.tolist()
        # activity = item
        # temperature = temperature.tolist()
        # humidity = humidity.tolist()
        activity = interpolate(activity)
        activity = np.asarray(activity)
        activity = np.divide(activity, herd_mean)

        print(len(activity), "%d/%d ..." % (cpt, len(X)))
        cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)

        if cpt == 0:
            X_cwt = pd.DataFrame(columns=[str(x) for x in range(len(cwt))], dtype=np.float16)

        X_cwt.loc[cpt] = cwt
        cpt += 1
        # print(X_cwt)

        target = data_frame.at[i, 'label']

        if target == 0:
            class0.append(cwt)
            class0_t.append(activity)
        if target == 1:
            class1.append(cwt)
            class1_t.append(activity)

        # if 'temperature' in out_fname:
        #     print('temperature')
        #     training_str_flatten = str(coefs.shape).strip('()') + \
        #                            ',' + str(cwt).strip('[]').replace(' ', '').replace('None', 'NaN') + \
        #                            ',' + str(temperature).strip('[]').replace(' ', '').replace('None', 'NaN') + \
        #                            ',' + str(humidity).strip('[]').replace(' ', '').replace('None', 'NaN') + \
        #                            ',' + \
        #                            str(meta).strip('[]').replace(' ', '').replace('None', 'NaN')
        # else:
        #     training_str_flatten = str(cwt).strip('[]').replace(' ', '').replace('None', 'NaN') + \
        #                            ',' + str(meta).strip('[]').replace(' ', '').replace('None', 'NaN')

        # print(" %s.....%s" % (training_str_flatten[0:50], training_str_flatten[-150:]))

        # outfile.write(training_str_flatten)
        # outfile.write('\n')

    # plt.hist(np.average(class0_t, axis=0), bins='auto')
    # plt.title("Histogram class0_t")
    # plt.show()
    #
    # plt.hist(np.average(class1_t, axis=0), bins='auto')
    # plt.title("Histogram class1_t")
    # plt.show()
    #
    # exit()

    class0_mean = np.average(class0_t, axis=0)
    _, coefs_class0_mean, _, _, _, _, _ = compute_cwt(class0_mean)
    class1_mean = np.average(class1_t, axis=0)
    _, coefs_class1_mean, _, _, _, _, _ = compute_cwt(class1_mean)

    # plt.bar(range(0, len(class0_mean)), class0_mean)
    # plt.show()
    # plt.pcolormesh(coefs_class0_mean, cmap='viridis')
    # plt.show()
    #
    # plt.bar(range(0, len(class1_mean)), class1_mean)
    # plt.show()
    # plt.pcolormesh(coefs_class1_mean, cmap='viridis')
    # plt.show()

    y = data_frame["label"].values.flatten()
    y = y.astype(int)
    X_cwt['label'] = y
    # results.append([X_cwt, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean])
    # class0_mean, class1_mean = [], []
    #
    # plot_cwt_coefs(coefs_class0_mean.shape[1], coefs_class0_mean, out_dir, out_fname, id='cwt_class0_mean',
    #                days=days, i=0, j=0)
    #
    # plot_cwt_coefs(coefs_class1_mean.shape[1], coefs_class1_mean, out_dir, out_fname, id='coefs_class1_mean',
    #                days=days, i=0, j=0)
    #
    # iwave0 = wavelet.icwt(coefs_class0_mean, scales, delta_t, wavelet=wavelet_type)
    # iwave0 = np.real(iwave0)
    # pot_icwt(iwave0, min(iwave0), max(iwave0), out_dir, out_fname, id='iwave0', days=days, i=i, j=0)
    #
    # iwave1 = wavelet.icwt(coefs_class1_mean, scales, delta_t, wavelet=wavelet_type)
    # iwave1 = np.real(iwave1)
    # pot_icwt(iwave1, min(iwave1), max(iwave1), out_dir, out_fname, id='iwave1', days=days, i=i, j=0)
    print("*********************************")
    print(data_frame)
    print(X_cwt)
    print("*********************************")
    return X_cwt, (scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean)


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr


def normalized(v):
    return v / np.sqrt(np.sum(v ** 2))


def interpolate_time(a, new_length):
    old_indices = np.arange(0, len(a))
    new_indices = np.linspace(0, len(a) - 1, new_length)
    spl = UnivariateSpline(old_indices, a, k=3, s=0)
    new_array = spl(new_indices)
    new_array[0] = 0
    return new_array


def plot_cwt_coefs(fig, axs, x, y, x_axis_lenght, coefs_class0_mean, out_dir, out_fname, id='', days=0, i=0, j=0, vmin_map=None, vmax_map=None, force_scale=True):
    # fig, ax = plt.subplots(figsize=(9, 4.8))
    ax = axs[x, y]
    ax.grid(False)

    if force_scale:
        vmin_map = coefs_class0_mean.min()
        vmax_map = coefs_class0_mean.max()

    time_axis = interpolate_time(np.arange(days+1), x_axis_lenght)
    im = ax.pcolormesh(time_axis, DATA_[0]['freqs'], coefs_class0_mean, cmap='viridis', vmin=vmin_map, vmax=vmax_map)
    fig.colorbar(im, ax=ax)
    ax.set_title('Magnitude')
    # fig.colorbar(im, ax=ax)
    ax.set_title('Continuous Wavelet Transform of %s on a %d days time period' % (id, days))
    ax.set_yscale('log')
    ax.set(xlabel="$Time (days)$", ylabel="$Frequency$(1/600th of an event per seconds)")
    # fig.show()
    pathlib.Path(out_dir+'\\cwt\\'+str(i)+'\\').mkdir(parents=True, exist_ok=True)
    outfile = '%s\\cwt\\%d\\cwt_%s_%d_%d_%s_%s_%s_days_%d_%s_%d_%d.png' % (
        out_dir,i, id, j, i, farm_id, out_fname.split('.')[0], resolution, days,
        str('empty').replace(',', '').replace('[', '').replace(']', ''),
        0, 0)
    # fig.savefig(outfile, dpi=100)
    # fig.clear()
    # plt.close(fig)


def pot_icwt(axs, x, y, iwave0, ymin2, ymax2, out_dir, out_fname, id='', days=0, i=0, j=0, force_scale=False):
    if force_scale or i in [0, 1, 2]:
        ymin2 = min(iwave0)
        ymax2 = max(iwave0)
    try:
        with plt.style.context("seaborn-white"):
            print("pot_icwt...")
            ax = axs[x, y]
            ax.grid(False)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            # fig, ax = plt.subplots(figsize=(25, 4.8))
            time_axis = interpolate_time(np.arange(days + 1), len(iwave0))
            ax.plot(time_axis, iwave0)
            # y_mean = [np.mean(iwave0)] * len(time_axis)
            # ax.plot(time_axis, y_mean, label='Mean', linestyle='--')
            del iwave0
            print([ymin2, ymax2])
            ax.set_ylim([ymin2, ymax2])
            # ax.set_title('Inverse Continuous Wavelet Transform of %s on a %d days time period' % (id, days))
            # ax.set(xlabel="$Time (days)$", ylabel="Activity (Summed sensor values over 10 minutes)")
            # ax.set(xlabel="$Time (days)$", ylabel="$Frequency (1/600th of an event per sec)$")
            # fig.show()
            # pathlib.Path(out_dir+'\\cwt\\'+str(i)+'\\').mkdir(parents=True, exist_ok=True)
            # outfile = '%s\\cwt\\%d\\%s_%d_%d_%s_%s_%s_days_%d_%s_%d_%d_.png' % (
            #     out_dir, i, id, j, i, farm_id, out_fname.split('.')[0], resolution, days,
            #     str('empty').replace(',', '').replace('[', '').replace(']', ''),
            #     0, 0)
            # fig.savefig(outfile, dpi=100)
            # fig.clear()
            # plt.close(fig)
    except ValueError as e:
        print(e)


def save_roc_curve(y_test, y_probas, title, options, folder, i=0, j=0):
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    # plt.title('ROC Curves %s' % title)
    split = title.split('\n')
    title = 'ROC Curves'
    skplt.metrics.plot_roc(y_test, y_probas, title=title, title_fontsize='medium')
    path = "%s/roc_curve/" % folder
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = '%s/%s' % (path, 'roc_%d_%d.png' % (j, i))
    final_path = final_path.replace('/', '\'').replace('\'', '\\').replace('\\', '/')
    print(final_path)
    plt.savefig(final_path)
    plt.show()
    plt.close()


def get_proba(y_probas, y_pred):
    class_0 = []
    class_1 = []
    for i, item in enumerate(y_probas):
        if y_pred[i] == 0:
            class_0.append(item[0])
        if y_pred[i] == 1:
            class_1.append(item[1])

    class_0 = np.asarray(class_0)
    class_1 = np.asarray(class_1)

    proba_0 = np.mean(class_0) if class_0.size > 0 else 0
    proba_1 = np.mean(class_1) if class_1.size > 0 else 0

    if np.isnan(proba_0):
        proba_0 = 0
    if np.isnan(proba_1):
        proba_1 = 0

    return proba_0 , proba_1


def next_multiple_of(x, n=40):
    return x + (n - x % n)


def lasso_feature_selection(X, y, shape, n_job=None):
    clf_l = LassoCV(cv=5, random_state=0, n_jobs=n_job, n_alphas=1).fit(X, y)
    weight_best_lasso = np.abs(clf_l.coef_)
    weight_best_lasso = minmax_scale(weight_best_lasso, feature_range=(0, 1))
    # weight_best_lasso = np.reshape(weight_best_lasso, shape)
    # returns map with location of important features 1 for important 0 for not important
    return weight_best_lasso


def rec_feature_selection(clf, X, y, n_features_to_select, shape, n_job=None):
    print("rec_feature_selection...")
    selector = RFECV(clf, step=1, cv=5, n_jobs=n_job, min_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)
    ranking = selector.ranking_
    # ranking = np.reshape(ranking, shape)
    ranking[ranking != 1] = 0
    # ranking = minmax_scale(ranking, feature_range=(0, 1))
    print("rec_feature_selection done")
    #returns map with location of important features 1 for important 0 for not important
    return ranking


def get_eli5_weight(aux1, i):
    class0 = aux1[aux1.target == aux1.target.unique()[i]]
    class0 = class0[class0.feature != '<BIAS>']
    class0['feature'] = class0['feature'].str.replace('x', '')
    class0['feature'] = class0['feature'].apply(int)
    class0 = class0.sort_values('feature')
    weight0 = class0['weight'].values
    return weight0


def get_weight_map_data(weight_array, shape, input, scales, delta_t, wavelet_type, force_abs=False):
    # input = minmax_scale(input, feature_range=(-1, 1))
    # weight_array = minmax_scale(weight_array, feature_range=(0, 1))
    weight_matrix = np.reshape(weight_array, shape)
    if force_abs:
        weight_matrix = np.abs(weight_matrix)
    if input is not None:
        weighted_input = np.multiply(weight_matrix, input)
    else:
        weighted_input = weight_matrix

    iwave = wavelet.icwt(weighted_input, scales, delta_t, wavelet=wavelet_type)
    iwave = np.real(iwave)
    if weighted_input[weighted_input == 0].size > 0:
        weighted_input[weighted_input == 0] = np.nan
    return weighted_input, iwave


def get_top_weight(weight, scale=2):
    n_to_keep = int(weight.shape[0] / scale)# keep half of best features
    print("n_to_keep=%d" % n_to_keep)
    top_features_indexes = weight.argsort()[-n_to_keep:][::-1]
    mask = np.zeros(weight.shape[0])
    mask[top_features_indexes] = weight[top_features_indexes]
    return mask


def get_min_max(data):
    cwt_list = []
    icwt_list = []
    for n, item in enumerate(data):
        if n in [0, 1, 2, 5, 6]: #todo fix exclude certain graphs ranges
            continue
        cwt_list.append(item[0].min())
        cwt_list.append(item[0].max())
        icwt_list.append(min(item[1]))
        icwt_list.append(max(item[1]))

    return min(cwt_list), max(cwt_list), min(icwt_list), max(icwt_list)


def explain_cwt(dfs, data, data_frame, data_frame_0,
                out_fname=None, df_hum=None, df_temp=None, resolution=None,
                days=None, f_config=None, farm_id=None, out_dir=None
                ):
    global DATA_
    plt.clf()
    print("process...", resolution, days)

    print("train_test_split...")

    for i, df in enumerate(dfs):
        scales, delta_t, wavelet_type, _, _, _, coefs_class0_mean, coefs_class1_mean, coefs_herd_mean = data[i]
        # df = shuffle(df)
        X = df[df.columns[0:df.shape[1] - 1]]
        y = df['label']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
        #                                                     random_state=int((datetime.now().microsecond)/10), stratify=y)

        # X = normalized(X)
        X_o = X.copy()
        for i in range(y[y == 0].shape[0]):
            # dummy_features = np.ones(X.shape[1])
            dummy_features = np.array(range(X.shape[1]))*0.0001
            # dummy_features = X_o.loc[i].values
            # dummy_features = np.random.rand(X.shape[1])
            X = np.vstack((X, dummy_features))
            dummy_target = 2
            y = np.append(y, dummy_target)


        X_train, X_test, y_train, y_test = X, X, y, y

        clf = SVC(kernel='linear', probability=True)
        #clf = LogisticRegression(C=1e10)

        # clf = LDA(n_components=2)
        # clf = Lasso(alpha=0.1)
        # clf = OneVsRestClassifier(LDA(n_components=3), n_jobs=3)

        print("finding best...")
        # weight_best_lasso = lasso_feature_selection(X, y, coefs_herd_mean.shape, n_job=-1)
        # weight_best_rec = rec_feature_selection(clf, X, y, int(X.loc[0].shape[0]/3), coefs_herd_mean.shape, n_job=-1)

        # weight_best_rec = weight_best_lasso

        print("fit...")
        clf.fit(X_train, y_train)

        X_lda, _, y_lda, _, clf_lda = reduce_lda(2, X, X, y, y)
        clf_lda.fit(X_lda, y_lda)
        # clf_svm = SVC(kernel='linear', probability=True)
        # clf_svm.fit(X_lda, y_lda)
        plot_2D_decision_boundaries_(X_lda, y_lda, X_lda, 'title', clf_lda, save=False)


        # clf = clf.best_estimator_
        y_pred = clf.predict(X_test)
        # print(classification_report(y_test, y_pred))

        # y_probas = clf.predict_proba(X_test)
        # y_probas = y_probas[:, :2]
        # save_roc_curve(y_test, y_probas, '', [], out_dir, i=i, j=v)

        # del X_train
        # del y_train
        # gc.collect()

        print("explain_prediction...")
        # print(eli5.explain_prediction_df(clf))
        # doc = np.average(X, axis=0)
        # aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(clf, doc)
        # aux1 = eli5.format_as_dataframe(aux1)
        # print("********************************")
        # print(aux1)
        # weight0 = get_eli5_weight(aux1, 0)
        # weight1 = get_eli5_weight(aux1, 1)
        # weight2 = get_eli5_weight(aux1, 2)

        # weight0 = np.multiply(doc, clf.coef_[0])
        # weight1 = np.multiply(doc, clf.coef_[1])
        # weight2 = np.multiply(doc, clf.coef_[2])

        weight0 = clf.coef_[0]
        weight1 = clf.coef_[1]
        weight2 = clf.coef_[2]

        pad_value = abs(np.prod(DATA_[0]['coef_shape']) - weight0.shape[0])
        for n in range(pad_value):
            weight0 = np.append(weight0, 0)
            weight1 = np.append(weight1, 0)
            weight2 = np.append(weight2, 0)


        weight0_best = get_top_weight(weight0)
        weight1_best = get_top_weight(weight1)

        weight0_best_2 = get_top_weight(weight0, scale=6)
        weight1_best_2 = get_top_weight(weight1, scale=6)
        weight0_best_3 = get_top_weight(weight0, scale=12)
        weight1_best_3 = get_top_weight(weight1, scale=12)

        print("building figure...")
        shape = DATA_[0]['coef_shape']
        data_to_plot = []
        data_to_plot.append((coefs_herd_mean,
                            np.real(wavelet.icwt(coefs_herd_mean, scales, delta_t, wavelet=wavelet_type))))
        data_to_plot.append((coefs_class0_mean,
                            np.real(wavelet.icwt(coefs_class0_mean, scales, delta_t, wavelet=wavelet_type))))
        data_to_plot.append((coefs_class1_mean,
                            np.real(wavelet.icwt(coefs_class1_mean, scales, delta_t, wavelet=wavelet_type))))

        data_to_plot.append(get_weight_map_data(weight0, shape, coefs_class0_mean, scales, delta_t, wavelet_type))
        data_to_plot.append(get_weight_map_data(weight0, shape, coefs_class1_mean, scales, delta_t, wavelet_type))


        data_to_plot.append(get_weight_map_data(weight0_best, shape, coefs_class0_mean, scales, delta_t, wavelet_type))
        data_to_plot.append(get_weight_map_data(weight1_best, shape, coefs_class1_mean, scales, delta_t, wavelet_type))
        # data_to_plot.append(get_weight_map_data(weight0_best_2, shape, coefs_class0_mean, scales, delta_t, wavelet_type))
        # data_to_plot.append(get_weight_map_data(weight1_best_2, shape, coefs_class1_mean, scales, delta_t, wavelet_type))
        data_to_plot.append(get_weight_map_data(weight0_best_3, shape, coefs_class0_mean, scales, delta_t, wavelet_type))
        data_to_plot.append(get_weight_map_data(weight1_best_3, shape, coefs_class1_mean, scales, delta_t, wavelet_type))


        x_axis = [x for x in range(int(weight0.size/scales.size))]

        w = len(x_axis)

        with plt.style.context("seaborn-white"):
            fig, axs = plt.subplots(len(data_to_plot), 2, facecolor='white')
            fig.set_size_inches(30, 5*len(data_to_plot))

            v_min_map, v_max_map, ymin2, ymax2 = get_min_max(data_to_plot)

            for i, item in enumerate(data_to_plot):
                plot_cwt_coefs(fig, axs, i, 0, w, item[0], out_dir, out_fname,  id='class0', days=days, i=i, j=0, vmin_map=v_min_map, vmax_map=v_max_map)
                pot_icwt(axs, i, 1, item[1], ymin2, ymax2, out_dir, out_fname, id='class0', days=days, i=i, j=0)


            fig.show()
            fig.savefig(out_dir+'\\'+'lassosubplot_%d.png' % 0, dpi=100, facecolor='white')
            # fig.clear()



            # gc.collect()
            # else:
            #     x_axis = [x for x in range(coefs_class1_mean.shape[1])]
            #     plot_cwt_coefs(x_axis, coefs_class1_mean, out_dir, out_fname, data_frame, f_config, id='class1', days=days)
            #
            #     c1 = np.reshape(normalized(weight1), DATA_[0]['coef_shape'])
            #     # del weight1
            #     print("computing icwt of weight1")
            #     iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
            #     iwave1 = np.real(iwave1)
            #     print(DATA_[0])
            #     print(iwave1)
            #
            #     ymin2 = min(iwave1)
            #     ymax2 = max(iwave1)
            #     pot_icwt(iwave1, ymin2, ymax2, out_dir, out_fname, data_frame, f_config, id='class1_i', days=days)


def slice_df(df):
    print(df['famacha_score'].value_counts())
    print(df)
    df = df.loc[:, :'label']
    np.random.seed(0)
    df = df.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.fillna(-1)
    df = shuffle(df)
    df['label'] = df['label'].map({True: 1, False: 0})
    print(df)
    return df


def get_mean_cwt(X):
    class0 = []
    for i, activity in enumerate(X):
        activity = interpolate(activity)
        activity = np.asarray(activity)
        class0.append(activity)
    class0 = np.average(class0, axis=0)
    _, coefs_class0, freqs, _, scales, _, _ = compute_cwt(class0)
    return coefs_class0


def process_data_frame(data_frame, data_frame_0, out_fname=None, df_hum=None, df_temp=None, resolution=None,
                       days=None):
    global DATA_
    DATA_ = []
    print(out_fname)
    #data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X_t = df_temp[df_temp.columns[0:df_temp.shape[1] - 1]].values
    X_h = df_hum[df_hum.columns[0:df_hum.shape[1] - 1]].values
    X_date = data_frame_0[data_frame_0.columns[data_frame_0.shape[1] - 7:data_frame_0.shape[1]]].values
    # cwt_list = []
    class0 = []
    class1 = []
    H = []
    # data_frame_0 = data_frame_0.reset_index(drop=True)

    for i, activity in enumerate(X):
        activity = interpolate(activity)
        activity = np.asarray(activity)
        H.append(activity)
    herd_mean = np.average(H, axis=0)

    del H
    gc.collect()
    print("finished computing herd mean.")

    # herd_mean = np.average(herd_data, axis=0)
    # herd_mean = interpolate(herd_mean)
    print("computing herd cwt")
    cwt_herd, coefs_herd_mean, freqs_h, _, _, _, _ = compute_cwt(herd_mean)
    DATA_.append({'coef_shape': coefs_herd_mean.shape, 'freqs': freqs_h})
    print("finished calculating herd cwt.")
    # herd_mean = minmax_scale(herd_mean, feature_range=(0, 1))

    purge_file(out_fname)
    X_cwt = pd.DataFrame()
    cpt = 0
    with open(out_fname, 'a') as outfile:
        for activity, (i, row), temperature, humidity in zip(X, data_frame_0.iterrows(), X_t, X_h):
            meta = row["label":].values.tolist()
            # activity = item
            temperature = temperature.tolist()
            humidity = humidity.tolist()
            activity = interpolate(activity)
            activity = np.asarray(activity)
            activity = np.divide(activity, herd_mean)

            # activity = minmax_scale(activity, feature_range=(0, 1))
            # if 'div' in out_fname:
            #     activity = np.divide(activity, herd_mean)
            #     print('div')
            # if 'sub' in out_fname:
            #     activity = np.subtract(activity, herd_mean)
            # activity[activity == np.inf] = np.nan
            # activity = interpolate(activity)

            print(len(activity), "%d/%d ..." % (i, len(X)))
            cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type = compute_cwt(activity)

            # if len(DATA_) < 1:
            #     DATA_.append({'coef_shape': coefs_herd_mean.shape, 'freqs': freqs_h})
            #     print(DATA_)

            if cpt == 0:
                X_cwt = pd.DataFrame(columns=[str(x) for x in range(len(cwt))], dtype=np.float16)

            X_cwt.loc[cpt] = cwt
            cpt += 1
            # print(X_cwt)
            target = data_frame.at[i, 'label']

            label = 'False'
            if target == 0:
                class0.append(activity)
            if target == 1:
                label = 'True'
                class1.append(activity)
            if 'temperature' in out_fname:
                print('temperature')
                training_str_flatten = str(coefs.shape).strip('()') + \
                                       ',' + str(cwt).strip('[]').replace(' ', '').replace('None', 'NaN') + \
                                       ',' + str(temperature).strip('[]').replace(' ', '').replace('None', 'NaN') + \
                                       ',' + str(humidity).strip('[]').replace(' ', '').replace('None', 'NaN') + \
                                       ',' + \
                                       str(meta).strip('[]').replace(' ', '').replace('None', 'NaN')
            else:
                training_str_flatten = str(cwt).strip('[]').replace(' ', '').replace('None', 'NaN') + \
                                       ',' + str(meta).strip('[]').replace(' ', '').replace('None', 'NaN')

            print(" %s.....%s" % (training_str_flatten[0:50], training_str_flatten[-150:]))

            # outfile.write(training_str_flatten)
            # outfile.write('\n')

    coefs_class0_mean = []
    class0_mean = []
    if len(class0) > 0:
        class0_mean = np.average(class0, axis=0)
        _, coefs_class0_mean, _, _, _, _, _ = compute_cwt(class0_mean)
    class1_mean = np.average(class1, axis=0)
    del class1
    _, coefs_class1_mean, _, _, _, _, _ = compute_cwt(class1_mean)
    X = X_cwt
    y = data_frame["label"].values.flatten()
    y = y.astype(int)
    return X.values, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean


# def process2(data_frame, data_frame_0, out_fname=None, df_hum=None, df_temp=None, resolution=None,
#             days=None, f_config=None, farm_id=None, out_dir=None, n_components=None):
#     global DATA_
#     plt.clf()
#     print("process...", resolution, days)
#     X, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, \
#     coefs_herd_mean, herd_mean = process_data_frame(
#         data_frame, data_frame_0, out_fname, df_hum=df_hum, df_temp=df_temp, days=days, resolution=resolution)
#     print("train_test_split...")
#
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     # X = normalize(X)
#     # X = preprocessing.MinMaxScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = X, X, y, y
#
#     # process_lda(X_train, y, 2)
#     clf = SVC(kernel='linear', C=1e10)
#     # clf = LDA(n_components=n_components, store_covariance=True)
#     # r = clf.fit_transform(X_train, y_test)
#
#     if n_components is None:
#         n_components = 0
#
#
#
#     # clf = LDA(n_components=2)
#
#     # y_train[0] = 3
#     # X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
#     # y_train = np.append(y_train, 3)
#
#     # X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
#     # y_test = np.append(y_test, 3)
#
#     # param_grid = {'C': np.logspace(-6, -1, 10), 'gamma': np.logspace(-6, -1, 10)}
#     # clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, n_jobs=4)
#
#     print("fit...")
#     clf.fit(X_train, y_train)
#     plt.plot(clf.coef_[0])
#     plt.plot(clf.coef_[1])
#     plt.plot(clf.coef_[2])
#     plt.show()
#
#     # clf = clf.best_estimator_
#     y_pred = clf.predict(X_test)
#     print(classification_report(y_test, y_pred))
#
#     del X_train
#     del y_train
#     gc.collect()
#
#     print("explain_prediction...")
#     aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(clf, X[0], top=X.shape[1])
#     aux1 = eli5.format_as_dataframe(aux1)
#     print("********************************")
#     print(aux1)
#     class0 = aux1[aux1.target == 0]
#     class1 = aux1[aux1.target == 1]
#     del aux1
#
#     class0 = class0[class0.feature != '<BIAS>']
#     class1 = class1[class1.feature != '<BIAS>']
#
#     class0['feature'] = class0['feature'].str.replace('x', '')
#     class1['feature'] = class1['feature'].str.replace('x', '')
#
#     class0['feature'] = class0['feature'].apply(int)
#     class1['feature'] = class1['feature'].apply(int)
#
#     class0 = class0.sort_values('feature')
#     class1 = class1.sort_values('feature')
#
#     weight0 = class0['weight'].values
#     weight1 = class1['weight'].values
#
#     del class1
#     weight0 = pad(weight0, DATA_[0]['coef_shape'][0] * DATA_[0]['coef_shape'][1])
#     weight1 = pad(weight1, DATA_[0]['coef_shape'][0] * DATA_[0]['coef_shape'][1])
#
#
#     print("building figure...")
#
#     if len(class0) > 0:
#         x_axis = [x for x in range(coefs_class0_mean.shape[1])]
#         plot_cwt_coefs(x_axis, coefs_class0_mean, out_dir, out_fname, data_frame, f_config, id='class0_%d' % n_components, days=days)
#         plot_cwt_coefs(x_axis, coefs_class1_mean, out_dir, out_fname, data_frame, f_config, id='class1_%d' % n_components, days=days)
#
#         c0 = np.reshape(weight0, DATA_[0]['coef_shape'])
#         del weight0
#         print("computing icwt of weight0")
#         iwave0 = wavelet.icwt(c0, scales, delta_t, wavelet=wavelet_type)
#         iwave0 = np.real(iwave0)
#         print(DATA_[0])
#         print(iwave0)
#
#         c1 = np.reshape(weight1, DATA_[0]['coef_shape'])
#         del weight1
#         print("computing icwt of weight1")
#         iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
#         iwave1 = np.real(iwave1)
#         print(DATA_[0])
#         print(iwave1)
#
#         ymin2 = min([min(iwave0), min(iwave1)])
#         ymax2 = max([max(iwave0), max(iwave1)])
#
#         pot_icwt(iwave0, ymin2, ymax2, out_dir, out_fname, data_frame, f_config, id='class0_%d' % n_components, days=days)
#         del iwave0
#
#         pot_icwt(iwave1, ymin2, ymax2, out_dir, out_fname, data_frame, f_config, id='class1_%d' % n_components, days=days)
#
#         gc.collect()
#         del coefs_class1_mean
#         del coefs_class0_mean
#     else:
#         x_axis = [x for x in range(coefs_class1_mean.shape[1])]
#         plot_cwt_coefs(x_axis, coefs_class1_mean, out_dir, out_fname, data_frame, f_config, id='class1', days=days)
#
#         c1 = np.reshape(normalized(weight1), DATA_[0]['coef_shape'])
#         del weight1
#         print("computing icwt of weight1")
#         iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
#         iwave1 = np.real(iwave1)
#         print(DATA_[0])
#         print(iwave1)
#
#         ymin2 = min(iwave1)
#         ymax2 = max(iwave1)
#         pot_icwt(iwave1, ymin2, ymax2, out_dir, out_fname, data_frame, f_config, id='class1_i', days=days)
#     DATA_ = []


if __name__ == '__main__':
    # try:
    #     shutil.rmtree("cedara_70091100056")
    # except (OSError, FileNotFoundError) as e:
    #     print(e)
    # try:
    #     shutil.rmtree("delmas_70101200027")
    # except (OSError, FileNotFoundError) as e:
    #     print(e)
    for resolution in ['10min']:
        # for item in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        #     dir = TRAINING_DIR + 'cedara_70091100056_resolution_%s_days_%d/' % (resolution, item)
        #
        #     try:
        #         start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_div.data', id=1)
        #         start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_.data', id=1)
        #     except MemoryError as e:
        #         print(e)
        #         DATA_ = []
        #         plt.clf()
        #         gc.collect()
        for farm_id in ["delmas_70101200027"]:
            # try:
            #     shutil.rmtree("transition")
            # except (OSError, FileNotFoundError) as e:
            #     print(e)

            for days in [7]:
                dir = TRAINING_DIR + '%s_sld_0_dbt%d_%s/' % (resolution, days, farm_id)
                # os.chdir(dir)

                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_resp_to_treat.data' % (days, farm_id),
                #       resolution=resolution,
                #       days=days,
                #       farm_id=farm_id,
                #       filter_delmas=(farm_id == 'delmas_70101200027'),
                #       filter_cedara=(farm_id == 'cedara_70091100056'),
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir,
                #       output_clf_transit=False
                #       )

                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_cwt_div.data' % (days, farm_id),
                #       resolution=resolution,
                #       days=days,
                #       farm_id=farm_id,
                #       filter_delmas=(farm_id == 'delmas_70101200027'),
                #       filter_cedara=(farm_id == 'cedara_70091100056'),
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir)
                #
                # exit()
                start(fname="%s/training_sets/activity_.data" % dir, out_fname='',
                      resolution=resolution,
                      days=days,
                      farm_id=farm_id+'_'+'kfold_100x10_hd',
                      filter_delmas=(farm_id == 'delmas_70101200027'),
                      filter_cedara=(farm_id == 'cedara_70091100056'),
                      fname_temp="%s/training_sets/temperature.data" % dir,
                      fname_hum="%s/training_sets/humidity.data" % dir,
                      output_clf_transit=True
                      )

                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_cwt_div.data' % (days, farm_id),
                #       resolution=resolution,
                #       days=days,
                #       farm_id='resp10_' + farm_id,
                #       filter_resp_to_treat_delmas=(farm_id == 'delmas_70101200027'),
                #       filter_resp_to_treat_cedara=(farm_id == 'cedara_70091100056'),
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir,
                #       output_clf_transit=False
                #       )
                # continue
                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_cwt_div.data' % (days, farm_id),
                #       resolution=resolution,
                #       days=days,
                #       farm_id='leave1out_'+farm_id,
                #       filter_delmas=(farm_id == 'delmas_70101200027'),
                #       filter_cedara=(farm_id == 'cedara_70091100056'),
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir,
                #       output_clf_transit=False
                #       )
                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_humidity_temperature_.data',
                #       resolution=resolution,
                #       days=days,
                #       farm_id=farm_id,
                #       filter_delmas=(farm_id == 'delmas_70101200027'),
                #       filter_cedara=(farm_id == 'cedara_70091100056'),
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir
                #       )

                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%s_cwt_div.data' % farm_id,
                #       resolution=resolution,
                #       f_config=[2, 1],
                #       days=days,
                #       farm_id=farm_id,
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir)
                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_sub.data',
                #       resolution=resolution,
                #       days=days,
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir)
                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_div.data',
                #       resolution=resolution,
                #       days=days,
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir)
                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='cwt_humidity_temperature_.data', id=1,
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir
                #       )
