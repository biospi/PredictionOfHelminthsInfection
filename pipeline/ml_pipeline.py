from __future__ import division  # for python2 regular div

import math
import shutil
# import pywt
import matplotlib.pyplot as plt
import matplotlib
import glob2
from sys import platform as _platform
from scipy import signal
if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sys import exit
import sys
import pathlib
from sklearn import datasets
import os
from sklearn.utils import shuffle
from multiprocessing import Pool
import random
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import unravel_index
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
import time
from sklearn.model_selection import cross_val_predict
import pycwt as wavelet
from sklearn.metrics import mean_squared_error, r2_score

META_DATA_LENGTH = 19
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from PIL import Image

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from astropy.convolution import convolve, Box1DKernel


class SelectKBestWrapper(SelectKBest):
    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X)


class PLSRegressionWrapper(PLSRegression):

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X)


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


def median_normalisation(a, m):
    a = a.astype(float)
    m = m.astype(float)
    sa = np.divide(a, m, out=np.zeros_like(a), where=m != 0)
    s = np.median(sa)
    mna = a * s
    return mna


def median_normalisation_p(activity_mean, activity):
    scale = np.zeros(len(activity))
    idx = []
    for n, a in enumerate(activity):
        if np.isnan(a) or a == 0 or np.isnan(activity_mean[n]):
            continue
        r = activity_mean[n] / a
        scale[n] = r
        idx.append(n)
    median = np.median(scale)
    # median = math.fabs(np.median(sorted(set(scale))))
    if median > 0:
        for i in idx:
            activity[i] = activity[i] * median
    return activity


def normalisation_l2(activity):
    activity_l2_norm = preprocessing.normalize(pd.DataFrame(activity.reshape(1, -1)))[0]
    return activity_l2_norm


def anscombe(value):
    return 2 * math.sqrt(abs(value) + (3 / 8))


def anscombe_list(activity):
    return [anscombe(x) if x is not None else None for x in activity]


def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def filter_by_entropy(df, thresh=2.5):
    filtered_samples = []
    for i in range(df.shape[0]-2):
        row = df.iloc[i, :-1]
        row_mean = df.iloc[i+1, :-1]
        row_median = df.iloc[i+2, :-1]
        target = df.iloc[i, -1]
        target_mean = df.iloc[i+1, -1]
        target_median = df.iloc[i+2, -1]
        if target == 'True' or target == 'False':
            h = entropy2(row)
            # print(h)
            if h > thresh:
                sample = row.values.tolist()
                sample.append(target)
                filtered_samples.append(sample)

                sample_mean = row_mean.values.tolist()
                sample_mean.append(target_mean)
                filtered_samples.append(sample_mean)

                sample_median = row_median.values.tolist()
                sample_median.append(target_median)
                filtered_samples.append(sample_median)

    df_filtered = pd.DataFrame(filtered_samples, columns=df.columns, dtype=float)
    return df_filtered


def get_norm_l2(data_frame_no_norm):
    """Apply l2 normalisation to each row in dataframe.

    Keyword arguments:
    data_frame_no_norm -- input raw dataframe containing samples (activity data, label/target)
    data_frame_mean -- mean dataframe containing median samples (mean activity data, label/target)
    """

    df_X_norm_l2 = pd.DataFrame(preprocessing.normalize(data_frame_no_norm.iloc[:, :-1]), columns=data_frame_no_norm.columns[:-1], dtype=float)
    df_X_norm_l2["label"] = data_frame_no_norm.iloc[:, -1]

    df_X_norm_l2_std = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df_X_norm_l2.iloc[:, :-1]), columns=data_frame_no_norm.columns[:-1], dtype=float)
    df_X_norm_l2_std["label"] = data_frame_no_norm.iloc[:, -1]

    return df_X_norm_l2_std


def get_median_norm_preprint(data_frame_no_norm, data_frame_mean):
    """Apply herd Median normalisation to each row in dataframe.

    Keyword arguments:
    data_frame_no_norm -- input raw dataframe containing samples (activity data, label/target)
    data_frame_mean -- mean dataframe containing median samples (mean activity data, label/target)
    """
    normalised_samples = []
    for i in range(data_frame_no_norm.shape[0]):
        activ = data_frame_no_norm.iloc[i, :-1].values
        mean = data_frame_mean.iloc[i, :-1].values
        label = data_frame_no_norm.iloc[i, -1]
        activ_norm = median_normalisation_p(mean, activ)
        sample = np.append(activ_norm, label)
        normalised_samples.append(sample)
    data_frame_median_norm = pd.DataFrame(normalised_samples, columns=data_frame_no_norm.columns, dtype=float)
    # data_frame_median_norm.fillna(0)
    return data_frame_median_norm


def get_median_norm(data_frame_no_norm, data_frame_median):
    """Apply herd Median normalisation to each row in dataframe.

    Keyword arguments:
    data_frame_no_norm -- input raw dataframe containing samples (activity data, label/target)
    data_frame_median -- median dataframe containing median samples (median activity data, label/target)
    """
    normalised_samples = []
    for i in range(data_frame_no_norm.shape[0]):
        activ = data_frame_no_norm.iloc[i, :-1].values
        median = data_frame_median.iloc[i, :-1].values
        label = data_frame_no_norm.iloc[i, -1]
        activ_norm = median_normalisation(activ, median)
        sample = np.append(activ_norm, label)
        normalised_samples.append(sample)
    data_frame_median_norm = pd.DataFrame(normalised_samples, columns=data_frame_no_norm.columns, dtype=float)
    return data_frame_median_norm


def get_anscombe(data_frame):
    """Apply anscomb transform to each row in dataframe.

    Keyword arguments:
    data_frame -- input dataframe containing samples (activity data, label/target)
    """
    anscombe_samples = []
    for i in range(data_frame.shape[0]):
        activ = data_frame.iloc[i, :-1].tolist()
        label = data_frame.iloc[i, -1]
        activ_anscomb = anscombe_list(activ)
        sample = np.append(activ_anscomb, label)
        anscombe_samples.append(sample)
    data_frame_anscomb = pd.DataFrame(anscombe_samples, columns=data_frame.columns, dtype=float)
    return data_frame_anscomb


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def plot_groups(graph_outputdir, df, title="title", xlabel='xlabel', ylabel='label', ntraces=1, idx_healthy=None, idx_unhealthy=None,
                show_max=True, show_min=False, show_mean=True, show_median=True, stepid=0):
    """Plot all rows in dataframe for each class Health or Unhealthy.

    Keyword arguments:
    df -- input dataframe containing samples (activity data, label/target)
    """
    df_healthy = df[df["label"] == False].iloc[:, :-1].values
    df_unhealthy = df[df["label"] == True].iloc[:, :-1].values

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(34.80, 7.20))
    fig.suptitle(title, fontsize=18)

    ymin = np.min(df.iloc[:, :-1].values)
    if idx_healthy is None or idx_unhealthy is None:
        ymax = np.max(df.iloc[:, :-1].values)
    else:
        ymax = max([np.max(df_healthy[idx_healthy]), np.max(df_unhealthy[idx_unhealthy])])

    if show_max:
        ymax = np.max(df_healthy)

    ticks = get_time_ticks(df_healthy.shape[1])

    if idx_healthy is None and ntraces is not None:
        idx_healthy = random.sample(range(1, df_healthy.shape[0]), ntraces)
    if ntraces is None:
        idx_healthy = list(range(df_healthy.shape[0]))
        idx_unhealthy = list(range(df_unhealthy.shape[0]))

    for i in idx_healthy:
        ax1.plot(ticks, df_healthy[i])
        ax1.set(xlabel=xlabel, ylabel=ylabel)
        if ntraces is None:
            ax1.set_title("Healthy animals %d / displaying %d" % (df_healthy.shape[0], df_healthy.shape[0]))
        else:
            ax1.set_title("Healthy animals %d / displaying %d" % (df_healthy.shape[0], ntraces))
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
    if idx_unhealthy is None:
        idx_unhealthy = random.sample(range(1, df_unhealthy.shape[0]), ntraces)
    for i in idx_unhealthy:
        ax2.plot(ticks, df_unhealthy[i])
        ax2.set(xlabel=xlabel, ylabel=ylabel)
        ax2.set_xticklabels(ticks, fontsize=12)
        if ntraces is None:
            ax2.set_title("Unhealthy animals %d / displaying %d" % (df_unhealthy.shape[0], df_unhealthy.shape[0]))
        else:
            ax2.set_title("Unhealthy animals %d / displaying %d" % (df_unhealthy.shape[0], ntraces))
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
    if show_max:
        # ax1.plot(ticks, np.amax(df_healthy, axis=0), c='tab:gray', label='max', linestyle='-')
        # ax2.plot(ticks, np.amax(df_unhealthy, axis=0), c='tab:gray', label='max', linestyle='-')
        ax1.fill_between(ticks, np.amax(df_healthy, axis=0), color='lightgrey', label='max', zorder=-1)
        ax2.fill_between(ticks, np.amax(df_unhealthy, axis=0), label='max', color='lightgrey')
        ax1.legend()
        ax2.legend()
    if show_min:
        ax1.plot(ticks, np.amin(df_healthy, axis=0), c='red', label='min')
        ax2.plot(ticks, np.amin(df_unhealthy, axis=0), c='red', label='min')
        ax1.legend()
        ax2.legend()

    if show_mean:
        ax1.plot(ticks, np.mean(df_healthy, axis=0), c='black', label='mean', alpha=1, linestyle='-')
        ax2.plot(ticks, np.mean(df_unhealthy, axis=0), c='black', label='mean', alpha=1, linestyle='-')
        ax1.legend()
        ax2.legend()

    if show_median:
        ax1.plot(ticks, np.median(df_healthy, axis=0), c='black', label='median', alpha=1, linestyle=':')
        ax2.plot(ticks, np.median(df_unhealthy, axis=0), c='black', label='median', alpha=1, linestyle=':')
        ax1.legend()
        ax2.legend()

    plt.show()
    filename = "%d_%s.png" % (stepid, title.replace(" ", "_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    print('saving fig...')
    fig.savefig(filepath)
    print("saved!")
    fig.clear()
    plt.close(fig)

    return idx_healthy, idx_unhealthy


def concatenate_images(im_dir, filter=None, title="title"):
    files = glob2.glob(im_dir)
    files = [file.replace("\\", '/') for file in files if "PCA" not in file]
    if filter is not None:
        files = [file.replace("\\", '/') for file in files if "cwt" in file]
    else:
        files = [file.replace("\\", '/') for file in files]
    images = [Image.open(x) for x in files]
    widths, heights = zip(*(i.size for i in images))

    total_h = sum(heights)
    max_w = max(widths)

    new_im = Image.new('RGB', (max_w, total_h))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    concat_impath = "%s/merge_%s.png" % (im_dir.replace("*.png", ""), title)
    new_im.save(concat_impath)


def downsample_df(data_frame):
    df_true = data_frame[data_frame['label'] == True]
    df_false = data_frame[data_frame['label'] == False]
    try:
        if df_false.shape[0] > df_true.shape[0]:
            df_false = df_false.sample(df_true.shape[0])
        else:
            print("more 12 to 11.")
            df_true = df_true.sample(df_false.shape[0])
    except ValueError as e:
        print(e)
        return
    data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)
    return data_frame


def load_df_from_datasets(enable_downsample_df, output_dir, fname, label_col='label', hi_pass_filter=None, low_pass_filter=None, n_process=None):
    print("load_df_from_datasets...", fname)
    # df = pd.read_csv(fname, nrows=1, sep=",", header=None, error_bad_lines=False)
    # # print(df)
    # type_dict = find_type_for_mem_opt(df)

    data_frame = pd.read_csv(fname, sep=",", header=None, low_memory=False)
    data_frame = data_frame.astype(dtype=float, errors='ignore')  # cast numeric values as float
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
    cols_to_keep = hearder[:-META_DATA_LENGTH]
    cols_to_keep.append(label_col)
    data_frame = data_frame[cols_to_keep]

    print(data_frame)

    # data_frame = shuffle(data_frame)

    # data_frame = filter_by_entropy(data_frame)

    # data_frame_no_norm = data_frame.loc[data_frame['label'].isin(["True", "False"])].reset_index(drop=True)
    # data_frame_no_norm = data_frame_no_norm.replace({"label": {'True': True, 'False': False}})
    data_frame_no_norm = data_frame
    enable_downsample_df = True
    if enable_downsample_df:
        data_frame_no_norm = downsample_df(data_frame_no_norm)

    data_frame_median = data_frame.loc[data_frame['label'].isin(["'median_True'", "'median_False'"])].reset_index(drop=True)
    data_frame_median = data_frame_median.replace({"label": {"'median_True'": "True", "'median_False'": "False"}})

    data_frame_mean = data_frame.loc[data_frame['label'].isin(["'mean_True'", "'mean_False'"])].reset_index(drop=True)
    data_frame_mean = data_frame_mean.replace({"label": {"'mean_True'": "True", "'mean_False'": "False"}})


    graph_outputdir = "%s/input_graphs/" % output_dir
    if os.path.exists(graph_outputdir):
        print("purge %s ..." % graph_outputdir)
        try:
            shutil.rmtree(graph_outputdir)
        except IOError:
            print("file not found.")
    create_rec_dir(graph_outputdir)
    ntraces = 2
    idx_healthy, idx_unhealthy = plot_groups(graph_outputdir, data_frame_no_norm, title="Raw thresholded samples", xlabel="Time",
                                             ylabel="activity", ntraces=ntraces)
    # idx_healthy = [1, 17]
    # idx_unhealthy = [47, 5]

    # plot_groups(graph_outputdir, data_frame_median, title="Median for each sample samples", xlabel="Time", ylabel="activity",
    #             idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=1)
    # plot_groups(graph_outputdir, data_frame_mean, title="Mean for each sample samples", xlabel="Time", ylabel="activity",
    #             idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=1)

    plot_time_pca(data_frame_no_norm, graph_outputdir, title="PCA time domain before normalisation")

    data_frame_median_norm = get_norm_l2(data_frame_no_norm)

    #data_frame_median_norm = get_median_norm_preprint(data_frame_no_norm, data_frame_mean)
    # data_frame_median_norm = get_median_norm(data_frame_no_norm, data_frame_median)

    plot_groups(graph_outputdir, data_frame_median_norm, title="Normalised samples", xlabel="Time", ylabel="activity",
                idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=2, ntraces=ntraces)

    plot_time_pca(data_frame_median_norm, graph_outputdir, title="PCA time domain after normalisation")

    data_frame_median_norm_anscombe = get_anscombe(data_frame_median_norm)

    plot_groups(graph_outputdir, data_frame_median_norm_anscombe, title="Normalisation and Anscombe for each sample samples",
                xlabel="Time",
                ylabel="activity", idx_healthy=idx_healthy, idx_unhealthy=idx_unhealthy, stepid=3, ntraces=ntraces)

    concatenate_images("%s/input_graphs/*.png" % output_dir, title="input_tramsform")

    data_frame_cwt_no_norm, _, _ = create_cwt_df(idx_healthy, idx_unhealthy, graph_outputdir, data_frame_no_norm.copy(),
                                                 title="Average cwt power of raw (no normalisation) samples", stepid=1,
                                                 hi_pass_filter=hi_pass_filter, low_pass=hi_pass_filter, ntraces=ntraces, n_process=n_process)
    data_frame_median_norm_cwt, _, _ = create_cwt_df(idx_healthy, idx_unhealthy, graph_outputdir, data_frame_median_norm.copy(),
                                                     title="Average cwt power of normalised samples",
                                                     stepid=2, hi_pass_filter=hi_pass_filter, low_pass=hi_pass_filter, ntraces=ntraces, n_process=n_process)
    data_frame_median_norm_cwt_anscombe, _, _ = create_cwt_df(idx_healthy, idx_unhealthy, graph_outputdir, data_frame_median_norm.copy(),
                                                              title="norm-cwt-ansc_Anscombe average cwt power of normalised samples",
                                                              stepid=3, enable_anscomb=True, hi_pass_filter=hi_pass_filter,
                                                              low_pass=hi_pass_filter, ntraces=ntraces, n_process=n_process)

    concatenate_images("%s/input_graphs/*.png" % output_dir, filter="cwt", title="spectogram_tranform")


    return data_frame_original, data_frame_no_norm, data_frame_median_norm, data_frame_median_norm_anscombe, \
           data_frame_cwt_no_norm, data_frame_median_norm_cwt, data_frame_median_norm_cwt_anscombe


def process_cross_farm(data_frame1, data_frame2, y_col='label'):
    print("process cross farm..")

    y1 = data_frame1[y_col].values.flatten()
    y1 = y1.astype(int)
    X1 = data_frame1[data_frame1.columns[2:data_frame1.shape[1] - 1]]

    y2 = data_frame2[y_col].values.flatten()
    y2 = y2.astype(int)
    X2 = data_frame2[data_frame2.columns[2:data_frame2.shape[1] - 1]]

    print("->SVC")
    pipe = Pipeline([('svc', SVC(probability=True, class_weight='balanced'))])
    pipe.fit(X1.copy(), y1.copy())
    y_pred = pipe.predict(X2.copy())
    print(classification_report(y2, y_pred))

    print("->StandardScaler->SVC")
    pipe = Pipeline(
        [('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
    pipe.fit(X1.copy(), y1.copy())
    y_pred = pipe.predict(X2.copy())
    print(classification_report(y2, y_pred))

    print("->MinMaxScaler->SVC")
    pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
    pipe.fit(X1.copy(), y1.copy())
    y_pred = pipe.predict(X2.copy())
    print(classification_report(y2, y_pred))


def purge_file(filename):
    print("purge %s..." % filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("file not found.")


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def dummy_run(X, y, test_size, filename):
    plt.show()
    print("dummy run!")
    X = pd.DataFrame(X)
    for i, row in X.iterrows():
        for j in row.index.values:
            X.at[i, j] = random.random()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y,
                                                        test_size=int(test_size) / 100)
    print("training", "class0=", y_train[y_train == 0].size, "class1=", y_train[y_train == 1].size)
    print("test", "class0=", y_test[y_test == 0].size, "class1=", y_test[y_test == 1].size)

    # plt.hist(X_train, bins='auto', histtype='step', density=True)
    # plt.title("Distribution of training data")
    # plt.show()
    purge_file(filename)
    with open(filename, 'a') as outfile:
        print("->SVC")
        pipe = Pipeline([('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))

        print("->LDA")
        pipe = Pipeline([('lda', LDA())])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))

        print("->StandardScaler->SVC")
        pipe = Pipeline(
            [('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))

        print("->MinMaxScaler->SVC")
        pipe = Pipeline(
            [('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        print(str(classification_report(y_test, y_pred, output_dict=True)))

        print("->StandardScaler->LDA(1)->SVC")
        pipe = Pipeline(
            [('scaler', preprocessing.StandardScaler()), ('lda', LDA(n_components=1)),
             ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))

        print("->LDA(1)->SVC")
        pipe = Pipeline([('reduce_dim', LDA(n_components=1)), ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))

    print("*******************************************")
    print("STEP BY STEP")
    print("*******************************************")

    clf_lda = LDA(n_components=1)
    X_train_r = clf_lda.fit_transform(X_train.copy(), y_train.copy())
    X_test_r = clf_lda.transform(X_test.copy())

    X_reduced = np.concatenate((X_train_r.copy(), X_test_r.copy()), axis=0)
    y_reduced = np.concatenate((y_train.copy(), y_test.copy()), axis=0)

    print("->LDA(1)->SVC")
    plot_2D_decision_boundaries(SVC(probability=True, class_weight='balanced'), "svc", "dim_reduc_name", 1, 1, "",
                                X_reduced.copy(),
                                y_reduced.copy(),
                                X_test_r.copy(),
                                y_test.copy(),
                                X_train_r.copy(),
                                y_train.copy())


def load_binary_iris():
    iris = datasets.load_iris()
    data_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                             columns=iris['feature_names'] + ['target'])
    data_iris = data_iris.drop_duplicates()
    data_iris = shuffle(data_iris)
    data_iris = data_iris[data_iris.target != 2.0]  # remove class 2
    X = data_iris[data_iris.columns[0:data_iris.shape[1] - 3]].values
    y = data_iris["target"].values.flatten()
    y = y.astype(int)
    return X, y


def load_binary_random():
    X, y = load_binary_iris()
    X = pd.DataFrame(X)
    for i, row in X.iterrows():
        for j in row.index.values:
            X.at[i, j] = random.random()
    return X, y


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


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


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            try:
                os.makedirs(dir_path)
            except FileExistsError as e:
                print(e)

def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig, thresh_i, thresh_z):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)
    label = r'Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )' % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r'Mean ROC (Mean AUC = %0.2f)' % mean_auc
    ax.plot(mean_fpr, mean_tpr, color='tab:blue',
            label=label,
            lw=2, alpha=.8)

    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #
    # confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    # ax.fill_between(mean_fpr, confidence_lower, confidence_upper, color='tab:blue', alpha=.2)
    #                 #label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic iteration")
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/interpol_%d_zero_%d/png/" % (out_dir, thresh_i, thresh_z)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)

    path = "%s/roc_curve/interpol_%d_zero_%d/svg/" % (out_dir, thresh_i, thresh_z)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    print(final_path)
    fig.savefig(final_path)


def make_roc_curve(out_dir, classifier, X, y, cv, param_str, thresh_i, thresh_z):
    print("make_roc_curve")

    if isinstance(X, pd.DataFrame):
        X = X.values
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.clf()
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             label=None,
                             alpha=0.3, lw=1, ax=ax, c="tab:blue")
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        # ax.plot(viz.fpr, viz.tpr, c="tab:green")
    clf_name = "%s_%s" % ("_".join([x[0] for x in classifier.steps]), param_str)
    plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, clf_name, fig, thresh_i, thresh_z)
    plt.close(fig)
    plt.clf()


def plot_2d_space_TSNE(X, y, filename_2d_scatter):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    plot_2d_space(tsne_results, y, filename_2d_scatter)


def plot_2d_space_cwt_scales(df_cwt_list, label='2D PCA of CWTS', y_col='label'):
    for i in range(df_cwt_list[0].shape[0]):
        plt.clf()
        fig, ax = plt.subplots(figsize=(12.80, 7.20))
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for item in df_cwt_list:
            item = item.replace({y_col: {'True': True, 'False': False}})
            X = item.iloc[i, :-1]
            y = item.iloc[i, -1]
            y = y.astype(int)
            for l, c, m in zip(np.unique(y), colors, markers):
                if l == 0:
                    ax.scatter(X.values[y == l, 0], X.values[y == l, 1], c=colors[0], alpha=0.4, marker=markers[0])
                if l == 1:
                    ax.scatter(X.values[y == l, 0], X.values[y == l, 1], c=colors[1], alpha=0.4, marker=markers[1])

        ax.set_title("PCA_SCALE_"+str(i)+"_"+label)
        # ax.legend(loc='upper right')
        # print(filename_2d_scatter)
        # fig.savefig(filename_2d_scatter)
        plt.show()
        # plt.close(fig)


def plot_2d_space(X, y, filename_2d_scatter, title='title'):
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    print("plot_2d_space", X[0])
    if len(X[0]) == 1:
        for l, c, m in zip(np.unique(y), colors, markers):
            ax.scatter(
                X[y == l, 0],
                np.zeros(X[y == l, 0].size),
                c=c, label=l, marker=m
            )
    else:
        for l, c, m in zip(np.unique(y), colors, markers):
            ax.scatter(
                X[y == l, 0],
                X[y == l, 1],
                c=c, label=l, marker=m
            )

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    print(filename_2d_scatter)
    folder = "/".join(filename_2d_scatter.split("/")[:-1])
    create_rec_dir(folder)
    fig.savefig(filename_2d_scatter)
    # plt.show()
    plt.close(fig)
    plt.clf()


def process_data_frame(out_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats, sampling,
                       downsample_false_class, y_col='label'):
    print("*******************************************************************")
    # print("downsample_false_class=", downsample_false_class)
    # print("*******************************************************************")
    # data_frame = data_frame.replace({y_col: {'True': True, 'False': False}})
    report_rows_list = []
    # if downsample_false_class:
    #     df_true = data_frame[data_frame['label'] == True]
    #     df_false = data_frame[data_frame['label'] == False]
    #     try:
    #         df_false = df_false.sample(df_true.shape[0])
    #     except ValueError as e:
    #         print(e)
    #         return
    #     data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)

    # data_frame = data_frame.dropna()
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]]

    # pls = PLSRegression(n_components=10)
    # pls.fit(X.copy(), y.copy())
    # y_cv = cross_val_predict(pls, X, y, cv=3)
    # score = r2_score(y, y_cv)
    # mse = mean_squared_error(y, y_cv)
    # print("r2score", score, "mse", mse)

    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)

    try:
        filename_2d_scatter = "%s/PCA/%s_2DPCA_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
            output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.copy())
        plot_2d_space(X_pca, y, filename_2d_scatter, '2 PCA components ' + option)


        filename_2d_scatter = "%s/ANOVA/%s_2DANOVA_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
            output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
        X_anova = SelectKBest(f_classif, k=2).fit_transform(X.copy(), y.copy())
        plot_2d_space(X_anova, y, filename_2d_scatter, '2 ANOVA components ' + option)

        # filename_2d_scatter = "%s/%s_1DLDA_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
        #     output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
        # lda = LDA(n_components=1)
        # X_lda = lda.fit_transform(X.copy(), y.copy())
        # plot_2d_space(X_lda, y, filename_2d_scatter, '(1 LDA components)')
        #
        # filename_2d_scatter = "%s/%s_2DTSNE_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
        #     output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
        # pca_50 = PCA(n_components=10)
        # X_pca50 = pca_50.fit_transform(X.copy())
        # plot_2d_space_TSNE(X_pca50, y, filename_2d_scatter)

        filename_2d_scatter = "%s/PLS/%s_2DPLS_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
            output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)

        pls = PLSRegression(n_components=2)
        X_pls= pls.fit_transform(X.copy(), y.copy())[0]
        plot_2d_space(X_pls, y, filename_2d_scatter, '2 PLS components ' + option)

    except ValueError as e:
        print(e)

    # X, y = load_binary_iris()
    # X, y = load_binary_random()

    # test_size = 10
    # print("test_size=", test_size, test_size / 100)
    # t_s = float(test_size / 100)
    # print("test size in percent=", t_s)
    # try:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=t_s)
    # except ValueError as e:
    #     print(e)
    #     return
    # print("training", "class0=", y_train[y_train == 0].size, "class1=", y_train[y_train == 1].size)
    # print("test", "class0=", y_test[y_test == 0].size, "class1=", y_test[y_test == 1].size)

    # plt.hist(X_train.values.flatten(), bins='auto', histtype='step', density=True)
    # plt.title("Distribution of training data")
    # plt.show()

    print("************************************************")
    print("downsample on= " + str(downsample_false_class))
    class0_count = str(y[y == 0].size)
    class1_count = str(y[y == 1].size)
    print("X-> class0=" + class0_count + " class1=" + class1_count)
    try:
        if int(class1_count) < 2 or int(class0_count) < 2:
            print("not enough samples!")
            return
    except ValueError as e:
        print(e)
        return

    scoring = {
        'balanced_accuracy_score': make_scorer(balanced_accuracy_score),
        'roc_auc_score': make_scorer(roc_auc_score, average=None),
        'precision_score0': make_scorer(precision_score, average=None, labels=[0]),
        'precision_score1': make_scorer(precision_score, average=None, labels=[1]),
        'recall_score0': make_scorer(recall_score, average=None, labels=[0]),
        'recall_score1': make_scorer(recall_score, average=None, labels=[1]),
        'f1_score0': make_scorer(f1_score, average=None, labels=[0]),
        'f1_score1': make_scorer(f1_score, average=None, labels=[1])
    }

    param_str = "option_%s_downsample_%s_threshi_%d_threshz_%d_days_%d_farmid_%s_nrepeat_%d_nsplits_%d_class0_%s_class1_%s_sampling_%s" % (
    option, str(downsample_false_class), thresh_i, thresh_z, days, farm_id, n_repeats, n_splits, class0_count,
    class1_count, sampling)

    # print('->PLSRegression')
    # clf_pls = make_pipeline(PLSRegression(n_components=2))
    # cv_pls = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_val_predict(clf_pls, X.copy(), y.copy(), n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->PLSRegression"
    # scores["classifier_details"] = str(clf_pls).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_pls, X, y, cv_pls, param_str, thresh_i, thresh_z)
    # del scores
    #
    #
    # print('->PLSRegression->SVC')
    # clf_pls_svc = make_pipeline(PLSRegression(n_components=2), SVC(probability=True, class_weight='balanced'))
    # cv_pls_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_pls_svc, X.copy(), y.copy(), cv=cv_pls_svc, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->PLSRegression->SVC"
    # scores["classifier_details"] = str(clf_pls_svc).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_pls_svc, X, y, cv_pls_svc, param_str, thresh_i, thresh_z)
    # del scores

    print('->SelectKBest(2)->SVM')
    clf_skb2_svm = make_pipeline(SelectKBest(f_classif, k=2), SVC(probability=True))
    cv_skb2_svm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    scores = cross_validate(clf_skb2_svm, X.copy(), y.copy(), cv=cv_skb2_svm, scoring=scoring, n_jobs=-1)
    scores["downsample"] = downsample_false_class
    scores["class0"] = y[y == 0].size
    scores["class1"] = y[y == 1].size
    scores["option"] = option
    scores["thresh_i"] = thresh_i
    scores["thresh_z"] = thresh_z
    scores["days"] = days
    scores["farm_id"] = farm_id
    scores["n_repeats"] = n_repeats
    scores["n_splits"] = n_splits
    scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    scores["sampling"] = sampling
    scores["classifier"] = "->SelectKBest(2)->SVM'"
    scores["classifier_details"] = str(clf_skb2_svm).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    clf_skb2_svm = make_pipeline(SelectKBest(f_classif, k=2), SVC(probability=True))
    cv_skb2_svm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    make_roc_curve(out_dir, clf_skb2_svm, X, y, cv_skb2_svm, param_str, thresh_i, thresh_z)
    del scores


    print('->PLS(2)->SVM')
    clf_pls2_svm = make_pipeline(PLSRegressionWrapper(n_components=2), SVC(probability=True, class_weight='balanced'))
    cv_pls2_svm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    scores = cross_validate(clf_pls2_svm, X.copy(), y.copy(), cv=cv_pls2_svm, scoring=scoring, n_jobs=-1)
    scores["downsample"] = downsample_false_class
    scores["class0"] = y[y == 0].size
    scores["class1"] = y[y == 1].size
    scores["option"] = option
    scores["thresh_i"] = thresh_i
    scores["thresh_z"] = thresh_z
    scores["days"] = days
    scores["farm_id"] = farm_id
    scores["n_repeats"] = n_repeats
    scores["n_splits"] = n_splits
    scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    scores["sampling"] = sampling
    scores["classifier"] = "->PLS(2)->SVM"
    scores["classifier_details"] = str(clf_pls2_svm).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    clf_pls2_svm = make_pipeline(PLSRegressionWrapper(n_components=2), SVC(probability=True, class_weight='balanced'))
    cv_pls2_svm = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    make_roc_curve(out_dir, clf_pls2_svm, X, y, cv_pls2_svm, param_str, thresh_i, thresh_z)
    del scores

    print('->SVC')
    clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    cv_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    scores = cross_validate(clf_svc, X.copy(), y.copy(), cv=cv_svc, scoring=scoring, n_jobs=-1)
    scores["downsample"] = downsample_false_class
    scores["class0"] = y[y == 0].size
    scores["class1"] = y[y == 1].size
    scores["option"] = option
    scores["thresh_i"] = thresh_i
    scores["thresh_z"] = thresh_z
    scores["days"] = days
    scores["farm_id"] = farm_id
    scores["n_repeats"] = n_repeats
    scores["n_splits"] = n_splits
    scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    scores["sampling"] = sampling
    scores["classifier"] = "->SVC"
    scores["classifier_details"] = str(clf_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    cv_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=0)
    make_roc_curve(out_dir, clf_svc, X, y, cv_svc, param_str, thresh_i, thresh_z)
    del scores



    # print('->LDA')
    # clf_lda = make_pipeline(LDA())
    # cv_lda = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                                  random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_lda, X.copy(), y.copy(), cv=cv_lda, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->LDA"
    # scores["classifier_details"] = str(clf_lda).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_lda, X, y, cv_lda, param_str, thresh_i, thresh_z)
    # del scores
    #
    # print('->LDA(1)->SVC')
    # clf_lda1_svc = make_pipeline(LDA(n_components=1), SVC(probability=True, class_weight='balanced'))
    # cv_lda1_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_lda1_svc, X.copy(), y.copy(), cv=cv_lda1_svc, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->LDA(1)->SVC"
    # scores["classifier_details"] = str(clf_lda1_svc).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_lda1_svc, X, y, cv_lda1_svc, param_str, thresh_i, thresh_z)
    # del scores

    # try:
    #     print('->PCA(50)->SVC')
    #     clf_pca50_svc = make_pipeline(PCA(n_components=50), SVC(probability=True, class_weight='balanced'))
    #     cv_pca50_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                                  random_state=int(datetime.now().microsecond / 10))
    #     scores = cross_validate(clf_pca50_svc, X.copy(), y.copy(), cv=cv_pca50_svc, scoring=scoring, n_jobs=-1)
    #     scores["downsample"] = downsample_false_class
    #     scores["class0"] = y[y == 0].size
    #     scores["class1"] = y[y == 1].size
    #     scores["option"] = option
    #     scores["thresh_i"] = thresh_i
    #     scores["thresh_z"] = thresh_z
    #     scores["days"] = days
    #     scores["farm_id"] = farm_id
    #     scores["n_repeats"] = n_repeats
    #     scores["n_splits"] = n_splits
    #     scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    #     scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    #     scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    #     scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    #     scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    #     scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    #     scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    #     scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    #     scores["sampling"] = sampling
    #     scores["classifier"] = "->PCA(50)->SVC"
    #     scores["classifier_details"] = str(clf_pca50_svc).replace('\n', '').replace(" ", '')
    #     report_rows_list.append(scores)
    #     make_roc_curve(out_dir, clf_pca50_svc, X, y, cv_pca50_svc, param_str, thresh_i, thresh_z)
    #     del scores
    # except ValueError as e:
    #     print(e)
    #

    print('->StandardScaler->SVC')
    clf_std_svc = make_pipeline(preprocessing.StandardScaler(), SVC(probability=True, class_weight='balanced'))
    cv_std_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                         random_state=0)
    scores = cross_validate(clf_std_svc, X.copy(), y.copy(), cv=cv_std_svc, scoring=scoring, n_jobs=-1)
    scores["downsample"] = downsample_false_class
    scores["class0"] = y[y == 0].size
    scores["class1"] = y[y == 1].size
    scores["option"] = option
    scores["thresh_i"] = thresh_i
    scores["thresh_z"] = thresh_z
    scores["days"] = days
    scores["farm_id"] = farm_id
    scores["n_repeats"] = n_repeats
    scores["n_splits"] = n_splits
    scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    scores["sampling"] = sampling
    scores["classifier"] = "->StandardScaler->SVC"
    scores["classifier_details"] = str(clf_std_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)

    clf_std_svc = make_pipeline(preprocessing.StandardScaler(), SVC(probability=True, class_weight='balanced'))
    cv_std_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                         random_state=0)
    make_roc_curve(out_dir, clf_std_svc, X, y, cv_std_svc, param_str, thresh_i, thresh_z)
    del scores

    # print('->MLP')
    # clf_mlp = make_pipeline(MLPClassifier(hidden_layer_sizes=(100, 10)))
    # cv_mlp = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                                  random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_mlp, X.copy(), y.copy(), cv=cv_mlp, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->MLP"
    # scores["classifier_details"] = str(clf_mlp).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # # make_roc_curve(out_dir, clf_mlp, X, y, cv_mlp, param_str, thresh_i, thresh_z)
    # del scores

    # print('->MinMaxScaler->SVC')
    # clf_minmax_svc = make_pipeline(preprocessing.MinMaxScaler(), SVC(probability=True, class_weight='balanced'))
    # cv_minmax_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_minmax_svc, X.copy(), y.copy(), cv=cv_minmax_svc, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->MinMaxScaler->SVC"
    # scores["classifier_details"] = str(clf_minmax_svc).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_minmax_svc, X, y, cv_minmax_svc, param_str, thresh_i, thresh_z)
    # del scores

    # print('->Normalize(l2)->SVC')
    # clf_normalize_svc = make_pipeline(preprocessing.Normalizer(), SVC(probability=True, class_weight='balanced'))
    # cv_normalize_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                                            random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_normalize_svc, X.copy(), y.copy(), cv=cv_normalize_svc, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->Normalize(l2)->SVC"
    # scores["classifier_details"] = str(clf_normalize_svc).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_normalize_svc, X, y, cv_normalize_svc, param_str, thresh_i, thresh_z)
    # del scores
    #
    # print('->Normalize(l2)->MinMaxScaler->SVC')
    # clf_normalize_minmax_svc = make_pipeline(preprocessing.Normalizer(), preprocessing.MinMaxScaler(), SVC(probability=True, class_weight='balanced'))
    # cv_normalize_minmax_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_normalize_minmax_svc, X.copy(), y.copy(), cv=cv_normalize_minmax_svc, scoring=scoring, n_jobs=-1)
    # scores["downsample"] = downsample_false_class
    # scores["class0"] = y[y == 0].size
    # scores["class1"] = y[y == 1].size
    # scores["option"] = option
    # scores["thresh_i"] = thresh_i
    # scores["thresh_z"] = thresh_z
    # scores["days"] = days
    # scores["farm_id"] = farm_id
    # scores["n_repeats"] = n_repeats
    # scores["n_splits"] = n_splits
    # scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    # scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    # scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    # scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    # scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    # scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    # scores["sampling"] = sampling
    # scores["classifier"] = "->Normalize(l2)->MinMaxScaler->SVC'"
    # scores["classifier_details"] = str(clf_normalize_minmax_svc).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_normalize_minmax_svc, X, y, cv_normalize_minmax_svc, param_str, thresh_i, thresh_z)
    # del scores

    print('->RandomForestClassifier')
    clf_normalize_random = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    cv_normalize_random = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=0)
    scores = cross_validate(clf_normalize_random, X.copy(), y.copy(), cv=cv_normalize_random, scoring=scoring, n_jobs=-1)
    scores["downsample"] = downsample_false_class
    scores["class0"] = y[y == 0].size
    scores["class1"] = y[y == 1].size
    scores["option"] = option
    scores["thresh_i"] = thresh_i
    scores["thresh_z"] = thresh_z
    scores["days"] = days
    scores["farm_id"] = farm_id
    scores["n_repeats"] = n_repeats
    scores["n_splits"] = n_splits
    scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
    scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
    scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
    scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
    scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
    scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
    scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
    scores["sampling"] = sampling
    scores["classifier"] = "->RandomForestClassifier"
    scores["classifier_details"] = str(clf_normalize_random).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    clf_normalize_random = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    cv_normalize_random = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=0)
    make_roc_curve(out_dir, clf_normalize_random, X, y, cv_normalize_random, param_str, thresh_i, thresh_z)
    del scores

    df_report = pd.DataFrame(report_rows_list)
    filename = "%s/%s_classification_report_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.csv" % (
        output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)
    df_report.to_csv(filename, sep=',', index=False)
    print("filename=", filename)

    #
    # exit(-1)
    # pipe = Pipeline([('svc', SVC(probability=True))])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, y_pred))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "SVC"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # print('->LDA')
    # pipe = Pipeline([('lda', LDA())])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, y_pred))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "LDA"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # print("->StandardScaler->SVC")
    # pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True))])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, np.round(y_pred)))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "StandardScaler->SVC"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # print("->MinMaxScaler->SVC")
    # pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True))])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, np.round(y_pred)))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "MinMaxScaler->SVC"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # print("->StandardScaler->LDA(1)->SVC")
    # pipe = Pipeline(
    #     [('scaler', preprocessing.StandardScaler()), ('lda', LDA(n_components=1)), ('svc', SVC(probability=True))])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, y_pred))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "->StandardScaler->LDA(1)->SVC"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # print("->LDA(1)->SVC")
    # pipe = Pipeline([('reduce_dim', LDA(n_components=1)), ('svc', SVC(probability=True))])
    # pipe.fit(X_train.copy(), y_train.copy())
    # y_pred = pipe.predict(X_test.copy())
    # print(classification_report(y_test, y_pred))
    # clf_r = classification_report(y_test, y_pred, output_dict=True)
    # report = {}
    # report["classifier"] = "->LDA(1)->SVC"
    # report['precision_0'] = clf_r['0']['precision']
    # report['recall_0'] = clf_r['0']['recall']
    # report['f1-score_0'] = clf_r['0']['f1-score']
    # report['support_0'] = clf_r['0']['support']
    # report['precision_1'] = clf_r['1']['precision']
    # report['recall_1'] = clf_r['1']['recall']
    # report['f1-score_1'] = clf_r['1']['f1-score']
    # report['support_1'] = clf_r['1']['support']
    # report["downsample"] = downsample_false_class
    # report["class0_training"] = y_train[y_train == 0].size
    # report["class1_training"] = y_train[y_train == 1].size
    # report["class0_testing"] = y_test[y_test == 0].size
    # report["class1_testing"] = y_test[y_test == 1].size
    # report["option"] = option
    # report["thresh_i"] = thresh_i
    # report["thresh_z"] = thresh_z
    # report["days"] = days
    # report["farm_id"] = farm_id
    # report_rows_list.append(report)
    #
    # df_report = pd.DataFrame(report_rows_list)
    # filename = "%s/%s_classification_report_days_%d_threshi_%d_threshz_%d_testsize_%d_%s.csv" % (
    # output_dir, farm_id, days, thresh_i, thresh_z, test_size, option)
    # if not os.path.exists(output_dir):
    #     print("mkdir", output_dir)
    #     os.makedirs(output_dir)
    # df_report.to_csv(filename, sep=',', index=False)
    # print("filename=", filename)


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


def plot_2D_decision_boundaries(model, clf_name, dim_reduc_name, dim, nfold, resolution, X_reduced, y_reduced, X_test_r,
                                y_test_r, X_train_r, y_train_r, n_bin=8):
    plt.clf()
    print('processing visualisation...')

    fig, ax = plt.subplots(figsize=(7., 4.8))

    min = abs(X_reduced.min()) + 1
    max = abs(X_reduced.max()) + 1
    step = float(np.max([min, max]) / 10)

    xx, yy = np.mgrid[-min:max:step, -min:max:step]
    grid = np.c_[xx.ravel(), yy.ravel()]
    if dim == 1:
        grid = np.c_[xx.ravel()]

    print("training...")
    print("nfeatures=%d" % X_train_r.shape[1], X_train_r.shape)
    model.fit(X_train_r.copy(), y_train_r.copy())

    y_pred_r = model.predict(X_test_r.copy())
    y_probas_r = model.predict_proba(X_test_r.copy())
    p_y_true, p_y_false = get_proba(y_probas_r, y_pred_r)
    acc = accuracy_score(y_test_r, y_pred_r)

    print("After reduction!")
    print(classification_report(y_test_r, y_pred_r))

    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, \
    support_false, support_true = get_prec_recall_fscore_support(y_test_r, y_pred_r)

    # print((clf_name, dim_reduc_name, dim, nfold, 0, acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
    #     p_y_false * 100, p_y_true * 100, np.count_nonzero(y_test_r == 0), np.count_nonzero(y_test_r == 1), np.count_nonzero(y_train_r == 0), np.count_nonzero(y_train_r == 1),
    #     np.count_nonzero(y_test_r == 0), np.count_nonzero(y_test_r == 1), resolution, ','.join([])))

    title = '%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, pb0=%d%%, pb1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s input=%s \n' % (
                clf_name, dim_reduc_name, dim, nfold, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false * 100, p_y_true * 100,
                np.count_nonzero(y_test_r == 0), np.count_nonzero(y_test_r == 1),
                np.count_nonzero(y_train_r == 0), np.count_nonzero(y_train_r == 1),
                np.count_nonzero(y_test_r == 0), np.count_nonzero(y_test_r == 1), resolution, ','.join([]))

    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    offset_r = 0
    offset_g = 0
    offset_b = 0
    colors = [((77 + offset_r) / 255, (157 + offset_g) / 255, (210 + offset_b) / 255),
              (1, 1, 1),
              ((255 + offset_r) / 255, (177 + offset_g) / 255, (106 + offset_b) / 255)]
    cm = LinearSegmentedColormap.from_list('name', colors, N=n_bin)

    for _ in range(0, 1):
        contour = ax.contourf(xx, yy, probs, n_bin, cmap=cm, antialiased=False, vmin=0, vmax=1, alpha=0.3, linewidth=0,
                              linestyles='dashed', zorder=-1)
        ax.contour(contour, cmap=cm, linewidth=1, linestyles='dashed', zorder=-1, alpha=1)

    ax_c = fig.colorbar(contour)

    ax_c.set_alpha(1)
    ax_c.draw_all()

    ax_c.set_label("$P(y = 1)$")

    X_reduced_0 = X_reduced[y_reduced == 0]
    X_reduced_1 = X_reduced[y_reduced == 1]

    X_reduced_0_t = X_test_r[y_test_r == 0]
    X_reduced_1_t = X_test_r[y_test_r == 1]

    marker_size = 150
    si = dim - 1
    ax.scatter(X_reduced_0_t[:, 0], X_reduced_0_t[:, si], c=(39 / 255, 111 / 255, 158 / 255), s=marker_size, vmin=-.2,
               vmax=1.2,
               edgecolor=(49 / 255, 121 / 255, 168 / 255), linewidth=0, marker='s', alpha=0.7, label='Class0 (Healthy)'
               , zorder=1)

    ax.scatter(X_reduced_1_t[:, 0], X_reduced_1_t[:, si], c=(251 / 255, 119 / 255, 0 / 255), s=marker_size, vmin=-.2,
               vmax=1.2,
               edgecolor=(255 / 255, 129 / 255, 10 / 255), linewidth=0, marker='^', alpha=0.7,
               label='Class1 (Unhealthy)'
               , zorder=1)

    ax.scatter(X_reduced_0_t[:, 0], X_reduced_0_t[:, si], s=marker_size - 10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', label='Test data', zorder=1)

    ax.scatter(X_reduced_1_t[:, 0], X_reduced_1_t[:, si], s=marker_size - 10, vmin=-.2, vmax=1.2,
               edgecolor="black", facecolors='none', zorder=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6, linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    handles, labels = ax.get_legend_handles_labels()
    db_line = Line2D([0], [0], color=(183 / 255, 37 / 255, 42 / 255), label='Decision boundary')
    handles.append(db_line)

    plt.legend(loc=2, fancybox=True, framealpha=0.4, handles=handles)

    ax.set_title(title)
    ttl = ax.title
    ttl.set_position([.57, 0.97])
    fig.show()
    plt.close()


def parse_param_from_filename(file):
    split = file.split("/")[-1].split('.')[0].split('_')
    thresh_i = int(split[-3])
    thresh_z = int(split[-1])
    sampling = split[-5]
    days = int(split[4])
    farm_id = split[1] + "_" + split[2]
    option = split[0]
    return thresh_i, thresh_z, days, farm_id, option, sampling


def mask_cwt(cwt, coi, scales, turn_off=False):
    if turn_off:
        return cwt
    print("masking cwt...")

    coi_line = []
    for j in range(cwt.shape[1]):
        for i, s in enumerate(scales):
            c = coi[j]
            if s > c:
                cwt[i:, j] = -1
                coi_line.append(i)
                break

    # plt.clf()
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.imshow(cwt)
    # ax.plot(coi_line, linestyle="--", linewidth=5, c="white")
    # ax.set_aspect('auto')
    # ax.set_yscale('log')
    # plt.show()

    # line = np.array(coi_line)
    # coi_line_array = max(line) - line

    return cwt, coi_line


def plot_time_pca(df_time_domain, output_dir, title="title", y_col="label"):
    # N = df_time_domain.shape[1]
    # plt.clf()
    # for i in range(df_time_domain.shape[0]):
    #     y = df_time_domain.iloc[i, :-1]
    #     target = df_time_domain.iloc[i, -1]
    #     yf = scipy.fftpack.fft(y)
    #     yf = np.abs(yf) ** 2
    #     color = "tab:orange" if target else "tab:blue"
    #     plt.plot(yf, c=color, alpha=0.1)
    # plt.show()

    df_time_pca = pd.DataFrame(PCA(n_components=2).fit_transform(df_time_domain.iloc[:, :-1]))
    df_time_pca["label"] = df_time_domain["label"]
    X = df_time_pca.iloc[:, :-1].values
    y = df_time_pca.iloc[:, -1].astype(int)
    filename = title.replace(" ", "_")
    filepath = "%s/%s.png" % (output_dir, filename)
    plot_2d_space(X, y, filepath, title=title)


def plot_cwt_power(df_fft, fftfreqs, fft_power, coi, activity, power_cwt_masked, power_cwt, coi_line_array, freqs, graph_outputdir, target, entropy, idx, title="title", time_domain_signal=None):
    df_healthy = df_fft[df_fft["label"] == False].iloc[:, :-1].values
    df_unhealthy = df_fft[df_fft["label"] == True].iloc[:, :-1].values

    plt.clf()
    fig, axs = plt.subplots(1, 3, figsize=(19.20, 7.20))
    fig.suptitle("Signal , CWT, FFT" + title, fontsize=18)

    # df_h = df_unhealthy
    health_status = "Unhealthy"
    if target == False:
        health_status = "Healthy"
        # df_h = df_healthy

    ticks = get_time_ticks(len(activity))
    axs[0].plot(ticks, activity, c="tab:orange" if target else "tab:blue")
    axs[0].set_title("Time domain signal " + health_status)
    axs[0].set(xlabel="Time", ylabel="activity")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[0].xaxis.set_major_locator(mdates.DayLocator())


    # print(power.max())
    # # levels = list(np.arange(0.001, power.max(), power.max()/1000))
    # levels = np.linspace(0.00001, power.max(), 10000).tolist()
    # N = len(activity)
    # t = np.arange(0, N)
    # period = 1 / freqs
    # dt = 1
    # axs[1].contourf(t, np.log2(period), np.log2(power, out=power, where=power > 0), np.log2(levels), extend='both')
    # axs[1].fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
    #                            t[:1] - dt, t[:1] - dt]),
    #         np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
    #                            np.log2(period[-1:]), [1e-9]]),
    #         'k', alpha=0.3, hatch='x')
    # Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
    #                            np.ceil(np.log2(period.max())))
    # axs[1].set_yticks(np.log2(Yticks))
    # axs[1].set_xlabel("Time")
    # axs[1].set_ylabel("Frequency [Hz]")
    # axs[1].set_ylim(axs[1].get_ylim()[::-1])
    # axs[1].set_title("CWT "+health_status)


    axs[1].imshow(power_cwt)
    axs[1].plot(coi_line_array, linestyle="--", linewidth=5, c="white")
    axs[1].set_aspect('auto')
    axs[1].set_title("CWT "+health_status)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Frequency of wavelet")
    # axs[1].set_yscale('log')
    n_x_ticks = axs[1].get_xticks().shape[0]
    labels = [item.strftime("%H:%M") for item in ticks]
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_x_ticks)))]
    labels_[0:2] = labels[0]
    labels_[-2:] = labels[0]
    axs[1].set_xticklabels(labels_)

    n_y_ticks = axs[1].get_yticks().shape[0]
    labels = ["%.4f" % item for item in freqs]
    # print(labels)
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    axs[1].set_yticklabels(labels_)

    # dt = 1
    # npts = power_cwt_masked.shape[1]
    # f_min = min(freqs)
    # f_max = max(freqs)
    # t = np.linspace(0, dt * npts, npts)
    # x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), power_cwt_masked.shape[0]))
    # axs[1].pcolormesh(ticks, y, power_cwt_masked, shading='auto')
    # axs[1].set_xlabel("Time")
    # axs[1].set_ylabel("Frequency [Hz]")
    # axs[1].set_yscale('log')
    # axs[1].set_ylim(f_min, f_max)
    # # axs[1].plot(ticks, np.log10(coi), linestyle="--", c="white")
    # axs[1].set_ylim(axs[1].get_ylim()[::-1])
    # axs[1].set_title("CWT "+health_status)
    # axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # axs[1].xaxis.set_major_locator(mdates.DayLocator())

    axs[2].plot(fftfreqs, fft_power, label="individual", c="tab:orange" if target else "tab:blue")
    axs[2].plot(fftfreqs, np.mean(df_unhealthy, axis=0), label="Mean of UnHealthy group", linestyle="--", c="tab:red")
    axs[2].plot(fftfreqs, np.mean(df_healthy, axis=0), label="Mean of Healthy group", linestyle="--", c="tab:green")

    # axs[2].plot(np.median(df_unhealthy, axis=0), label="Median of UnHealthy group", linestyle=":", c="tab:red")
    # axs[2].plot(np.median(df_healthy, axis=0), label="Median of Healthy group", linestyle=":", c="tab:green")

    # axs[2].plot(np.mean(df_unhealthy, axis=0), label="Mean of UnHealthy group", linestyle="--", c="tab:red")
    # axs[2].plot(np.median(df_healthy, axis=0), label="Mean of Healthy group", linestyle=":", c="tab:green")

    axs[2].set_ylabel("log Amplitude")
    axs[2].set_xlabel("Frequency (event / 1minute)")
    axs[2].set_yscale('log')
    axs[2].set_title("FFT " + health_status)
    axs[2].legend()

    # plt.show()
    filename = "%s_%d_cwt.png" % (target, idx)
    filepath = "%s/%s/" % (graph_outputdir, title.replace(" ", "_").replace("(", "").replace(")", ""))
    create_rec_dir(filepath)
    print('saving fig...')
    fig.savefig(filepath+filename)
    print("saved!")
    fig.clear()
    plt.close(fig)


def plot_cwt_pca(df_cwt, title, graph_outputdir, stepid=5, xlabel="CWT Frequency index", ylabel="PCA component",
                 show_min=True, show_max=True, show_mean=True, show_median=True):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.80, 7.20))
    fig.suptitle("PCA_"+title, fontsize=18)
    ymax = np.max(df_cwt.iloc[:, :-1].values)
    ymin = np.min(df_cwt.iloc[:, :-1].values)
    for i in range(df_cwt.shape[0]):
        trace = df_cwt.iloc[i, :-1].tolist()
        color = "tab:orange" if df_cwt.iloc[i, -1] else "tab:blue"
        if color == "tab:orange":
            continue
        ax1.plot(trace, c=color, alpha=0.1)
        ax1.set(xlabel=xlabel, ylabel=ylabel)
        ax1.set_title("Healthy animals")
        ax1.set_ylim([ymin, ymax])

    for i in range(df_cwt.shape[0]):
        trace = df_cwt.iloc[i, :-1].tolist()
        color = "tab:orange" if df_cwt.iloc[i, -1] else "tab:blue"
        if color == "tab:blue":
            continue
        ax2.plot(trace, c=color, alpha=0.1)
        ax2.set(xlabel=xlabel, ylabel=ylabel)
        ax2.set_title("UnHealthy animals")
        ax2.set_ylim([ymin, ymax])

    df_healthy = df_cwt[df_cwt["label"] == False].iloc[:, :-1].values
    df_unhealthy = df_cwt[df_cwt["label"] == True].iloc[:, :-1].values
    if show_max:
        ax1.plot(np.amax(df_healthy, axis=0), c='black', label='max', alpha=1)
        ax2.plot(np.amax(df_unhealthy, axis=0), c='black', label='max', alpha=1)
        ax1.legend()
        ax2.legend()
    if show_min:
        ax1.plot(np.amin(df_healthy, axis=0), c='red', label='min', alpha=1)
        ax2.plot(np.amin(df_unhealthy, axis=0), c='red', label='min', alpha=1)
        ax1.legend()
        ax2.legend()
    if show_mean:
        ax1.plot(np.mean(df_healthy, axis=0), c='black', label='mean', alpha=1, linestyle='--')
        ax2.plot(np.mean(df_unhealthy, axis=0), c='black', label='mean', alpha=1, linestyle='--')
        ax1.legend()
        ax2.legend()

    if show_median:
        ax1.plot(np.median(df_healthy, axis=0), c='black', label='median', alpha=1, linestyle=':')
        ax2.plot(np.median(df_unhealthy, axis=0), c='black', label='median', alpha=1, linestyle=':')
        ax1.legend()
        ax2.legend()


    plt.show()
    filename = "%d_%s.png" % (stepid, title.replace(" ", "_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    print('saving fig...')
    fig.savefig(filepath)
    print("saved!")
    fig.clear()
    plt.close(fig)


def plot_cwt_power_sidebyside(idx_healthy, idx_unhealthy, coi_line_array, df_timedomain, graph_outputdir, power_masked_healthy, power_masked_unhealthy, freqs, ntraces=3, title="title", stepid=10):
    total_healthy = df_timedomain[df_timedomain["label"] == False].shape[0]
    total_unhealthy = df_timedomain[df_timedomain["label"] == True].shape[0]
    plt.clf()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.80, 7.20))
    fig.suptitle(title, fontsize=18)

    df_healthy = df_timedomain[df_timedomain["label"] == False].iloc[:, :-1].values
    df_unhealthy = df_timedomain[df_timedomain["label"] == True].iloc[:, :-1].values
    # ymin = 0
    # ymax = max([np.max(df_healthy), np.max(df_unhealthy)])
    #
    # idx_healthy = range(df_healthy.shape[0])
    # idx_unhealthy = range(df_unhealthy.shape[0])

    ymin = np.min(df_timedomain.iloc[:, :-1].values)
    if idx_healthy is None or idx_unhealthy is None:
        ymax = np.max(df_timedomain.iloc[:, :-1].values)
    else:
        ymax = max([np.max(df_healthy[idx_healthy]), np.max(df_unhealthy[idx_unhealthy])])

    # if show_max:
    #     ymax = np.max(df_healthy)

    ticks = get_time_ticks(df_healthy.shape[1])

    if idx_healthy is None and ntraces is not None:
        idx_healthy = random.sample(range(1, df_healthy.shape[0]), ntraces)
    if ntraces is None:
        idx_healthy = list(range(df_healthy.shape[0]))
        idx_unhealthy = list(range(df_unhealthy.shape[0]))

    for i in idx_healthy:
        ax1.plot(ticks, df_healthy[i])
        ax1.set(xlabel="Time", ylabel="activity")
        ax1.set_title("Healthy animals %d / displaying %d" % (total_healthy, len(idx_healthy)))
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())

    for i in idx_unhealthy:
        ax2.plot(ticks, df_unhealthy[i])
        ax2.set(xlabel="Time", ylabel="activity")
        ax2.set_yticks(ax2.get_yticks().tolist())
        ax2.set_xticklabels(ticks, fontsize=12)
        ax2.set_title("Unhealthy animals %d / displaying %d" % (total_unhealthy, len(idx_unhealthy)))
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())


    # dt = 1
    # npts = power_masked_healthy.shape[1]
    # f_min = min(freqs)
    # f_max = max(freqs)
    # t = np.linspace(0, dt * npts, npts)
    # x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), power_masked_healthy.shape[0]))
    # ax3.set_title("Healthy animals elem wise average of %d cwts" % df_healthy.shape[0])
    # ax3.pcolormesh(ticks, y, power_masked_healthy, shading='auto')
    # ax3.set_xlabel("Time")
    # ax3.set_ylabel("Frequency [Hz]")
    # ax3.set_yscale('log')
    # ax3.set_ylim(f_min, f_max)
    # ax3.set_ylim(ax3.get_ylim()[::-1])
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax3.xaxis.set_major_locator(mdates.DayLocator())

    ax3.imshow(power_masked_healthy)
    ax3.plot(coi_line_array, linestyle="--", linewidth=3, c="white")
    ax3.set_aspect('auto')
    ax3.set_title("Healthy animals elem wise average of %d cwts" % df_healthy.shape[0])
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Frequency")
    # ax3.set_yscale('log')
    n_x_ticks = ax3.get_xticks().shape[0]
    labels = [item.strftime("%H:%M") for item in ticks]
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_x_ticks)))]
    labels_[0:2] = labels[0]
    labels_[-2:] = labels[0]
    ax3.set_xticklabels(labels_)

    n_y_ticks = ax3.get_yticks().shape[0]
    labels = ["%.4f" % item for item in freqs]
    # print(labels)
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    ax3.set_yticklabels(labels_)

    # ax4.set_title("Unhealthy animals elem wise average of %d cwts" % df_healthy.shape[0])
    # ax4.pcolormesh(ticks, y, power_masked_unhealthy, shading='auto')
    # ax4.set_xlabel("Time")
    # ax4.set_ylabel("Frequency [Hz]")
    # ax4.set_yscale('log')
    # ax4.set_ylim(f_min, f_max)
    # ax4.set_ylim(ax4.get_ylim()[::-1])
    # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.imshow(power_masked_unhealthy)
    ax4.plot(coi_line_array, linestyle="--", linewidth=3, c="white")
    ax4.set_aspect('auto')
    ax4.set_title("Unhealthy animals elem wise average of %d cwts" % df_healthy.shape[0])
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Frequency")
    # ax4.set_yscale('log')
    n_x_ticks = ax4.get_xticks().shape[0]
    labels = [item.strftime("%H:%M") for item in ticks]
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_x_ticks)))]
    labels_[0:2] = labels[0]
    labels_[-2:] = labels[0]
    ax4.set_xticklabels(labels_)

    n_y_ticks = ax4.get_yticks().shape[0]
    labels = ["%.4f" % item for item in freqs]
    # print(labels)
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    ax4.set_yticklabels(labels_)

    plt.show()
    filename = "%d_%s.png" % (stepid, title.replace(" ","_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    print('saving fig...')
    fig.savefig(filepath)
    print("saved!")
    fig.clear()
    plt.close(fig)


# def compute_cwt_group(activity, target, i, total, low_pass, high_pass, pca_n_components):
#     print("%d/%d" % (i, total))
#     wavelet_type = 'morlet'
#     y = activity
#     coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1, dj=1/20, wavelet=wavelet_type)
#     coefs_cc = np.conj(coefs)
#     power = np.real(np.multiply(coefs, coefs_cc))
#     power_masked = mask_cwt(power.copy(), coi)
#
#     # if high_pass is not None and high_pass > 0:
#     #     power_masked = power_masked[low_pass:-high_pass, :]
#
#     data_pca = PCA(n_components=pca_n_components).fit_transform(power_masked).reshape(1, -1).tolist()[0]
#     data_pca.append(target)
#     return data_pca


def compute_fft_group(activity, target, i, total):
    print("%d/%d" % (i, total))
    fft = np.fft.fftshift(np.fft.fft(activity))
    fft_cc = np.conj(fft)
    power_fft = np.real(np.multiply(fft, fft_cc)).tolist()
    # smooth_data = convolve(power_fft, kernel=Box1DKernel(31))
    # power_fft = smooth_data.tolist()
    power_fft.append(target)
    return power_fft


def get_n_largest_coefs_fft(matrix, n=50):
    # matrix[matrix == -1] = np.nan
    features_list = []
    for i in range(n):
        location = matrix.argmax()
        value = matrix[location]
        features = [location, value]
        matrix[location] = -1
        features_list.append(features)
    f_array = np.array(features_list).flatten().tolist()
    # plt.clf()
    # plt.title("get_n_largest_coefs")
    # plt.imshow(matrix, aspect='auto')
    # plt.show()
    return f_array


def get_n_largest_coefs(matrix, n=50):
    # matrix[matrix == -1] = np.nan
    features_list = []
    for i in range(n):
        location = unravel_index(matrix.argmax(), matrix.shape)
        value = matrix[location]
        features = [location[0], location[1], value]
        matrix[location] = -1
        features_list.append(features)
    f_array = np.array(features_list).flatten().tolist()
    # plt.clf()
    # plt.title("get_n_largest_coefs")
    # plt.imshow(matrix, aspect='auto')
    # plt.show()
    return f_array


def compute_cwt(df_fft, activity, target, i, total,low_pass, high_pass, pca_n_components, graph_outputdir, title):
    print("%d/%d" % (i, total))
    # wavelet_type = 'morlet'
    y = activity
    w = wavelet.Morlet(3)
    coefs, scales, freqs, coi, _, _ = wavelet.cwt(y, 1, wavelet=w)
    # scales = range(len(activity))
    # coefs, freqs = pywt.cwt(y, scales, 'morl', 1)

    fft = np.fft.fftshift(np.fft.fft(activity))
    fft_cc = np.conj(fft)
    power_fft = np.real(np.multiply(fft, fft_cc))
    # power_fft = convolve(power_fft, kernel=Box1DKernel(31))
    n = len(activity)
    timestep = 1
    fftfreqs = np.fft.fftfreq(n, d=timestep)
    fftfreqs_ = []
    for f in fftfreqs:
        if f < 0:
            fftfreqs_.append(f)
    fftfreqs_.append(0.0)
    for f in fftfreqs:
        if f > 0:
            fftfreqs_.append(f)

    coefs_cc = np.conj(coefs)
    power_cwt = np.real(np.multiply(coefs, coefs_cc))
    power_masked, coi_line_array = mask_cwt(power_cwt.copy(), coi, scales)

    if high_pass is not None and high_pass > 0:
        power_masked = power_masked[low_pass:-high_pass, :]

    # data_pca = PCA(n_components=pca_n_components).fit_transform(power_masked).reshape(1, -1).tolist()[0]
    # data_pca.append(target)

    plot_cwt_power(df_fft, fftfreqs_, power_fft, coi, activity, power_masked,
                   power_cwt.copy(), coi_line_array, freqs, graph_outputdir, target,
                   entropy2(power_masked.flatten()[power_masked.flatten()>0]), i, title=title)

    # df_data_pca = pd.DataFrame(PCA(n_components=pca_n_components).fit_transform(power_masked))
    # df_data_pca["label"] = target

    power_flatten_masked = []
    for c in power_masked.flatten():
        if c == -1:
            continue
        power_flatten_masked.append(c)

    #max_coef_features = get_n_largest_coefs(power_masked.copy(), n=int(power_masked.size / 2))
    # max_coef_features_fft = get_n_largest_coefs_fft(power_fft.copy(), n=int(power_masked.size / 2))

    print("power_flatten_len=", len(power_flatten_masked))
    data = power_flatten_masked
    #data = power_flatten_masked + max_coef_features
    #data = max_coef_features
    # data = np.random.randint(1000, size=len(power_flatten_masked)).tolist()
    # data = [min(power_flatten_masked), max(power_flatten_masked), np.mean(power_flatten_masked), np.median(power_flatten_masked)]
    # data = power_flatten_masked + [min(power_flatten_masked), max(power_flatten_masked), np.mean(power_flatten_masked), np.median(power_flatten_masked)]
    # data = power_fft.tolist() + max_coef_features_fft
    #data = power_fft.tolist()
    # data = power_flatten_masked + power_fft.tolist()
    #data = power_fft[int(len(power_fft)/2):].tolist()
    data.append(target)

    # return [data_pca, power_masked, target, freqs, df_data_pca]

    return [data, power_cwt, target, freqs, coi_line_array]


def normalized(v):
    return v / np.sqrt(np.sum(v ** 2))


def create_cwt_df(idx_healthy, idx_unhealthy, graph_outputdir, df_timedomain, title="title", stepid=0, enable_anscomb=False,
                  low_pass=None, hi_pass_filter=None, pca_n_components=1, ntraces=None, n_process=6):

    # pool_cwt_group = Pool(processes=6)
    # results_cwt_pca_group = []
    # for i, row in enumerate(df_timedomain.iterrows()):
    #     target = row[1][-1]
    #     activity = row[1][0:-1].values
    #     results_cwt_pca_group.append(pool_cwt_group.apply_async(compute_cwt_group,
    #                                                             (activity, target, i, df_timedomain.shape[0],
    #                                                              low_pass, hi_pass_filter, pca_n_components,)))
    #
    # pool_cwt_group.close()
    # pool_cwt_group.join()
    # pool_cwt_group.terminate()
    #
    # pca_cwt_list = []
    # for res in results_cwt_pca_group:
    #     pca_cwt = res.get()
    #     pca_cwt_list.append(pca_cwt)
    # df_pca1_cwt = pd.DataFrame(pca_cwt_list)
    # colums = [str(x) for x in range(df_pca1_cwt.shape[1])]
    # colums[-1] = "label"
    # df_pca1_cwt.columns = colums

    pool_fft_group = Pool(processes=n_process)
    results_fftgroup = []
    for i, row in enumerate(df_timedomain.iterrows()):
        target = row[1][-1]
        activity = row[1][0:-1].values
        results_fftgroup.append(pool_fft_group.apply_async(compute_fft_group, (activity, target, i, df_timedomain.shape[0],)))

    pool_fft_group.close()
    pool_fft_group.join()
    pool_fft_group.terminate()

    fft_list = []
    for res in results_fftgroup:
        fft_ = res.get()
        fft_list.append(fft_)
    df_fft = pd.DataFrame(fft_list)
    colums = [str(x) for x in range(df_fft.shape[1])]
    colums[-1] = "label"
    df_fft.columns = colums

    pool_cwt = Pool(processes=n_process)
    results_cwt = []
    results_cwt_matrix_healthy = []
    results_cwt_matrix_unhealthy = []
    for i, row in enumerate(df_timedomain.iterrows()):
        target = row[1][-1]
        activity = row[1][0:-1].values
        results_cwt.append(pool_cwt.apply_async(compute_cwt, (df_fft, activity, target, i, df_timedomain.shape[0], low_pass, hi_pass_filter, pca_n_components, graph_outputdir, title,)))

    pool_cwt.close()
    pool_cwt.join()
    pool_cwt.terminate()

    features_flatten_sample_list = []
    freqs = None
    pca_healthy = []
    pca_unhealthy = []
    # df_cwt_pca_per_scale_list = []
    for res in results_cwt:
        features_flatten_sample = res.get()[0]
        features_flatten_sample_list.append(features_flatten_sample)
        power_matrix = res.get()[1]
        power_target = res.get()[2]
        freqs = res.get()[3]
        coi_line_array = res.get()[4]

        # df_cwt_pca_per_scale = res.get()[4]
        # df_cwt_pca_per_scale_list.append(df_cwt_pca_per_scale)
        if power_target:
            results_cwt_matrix_healthy.append(power_matrix)
            pca_healthy.append(features_flatten_sample)
        else:
            results_cwt_matrix_unhealthy.append(power_matrix)
            pca_unhealthy.append(features_flatten_sample)

    # plot_2d_space_cwt_scales(df_cwt_pca_per_scale_list, label=title)

    # plot_cwt_power(results_cwt_matrix_healthy[0], freqs)
    # plot_cwt_power(results_cwt_matrix_unhealthy[0], freqs)
    h_m = np.mean(results_cwt_matrix_healthy, axis=0)
    uh_m = np.mean(results_cwt_matrix_unhealthy, axis=0)
    if enable_anscomb:
        h_ma = get_anscombe(pd.DataFrame(h_m)).values
        uh_ma = get_anscombe(pd.DataFrame(uh_m)).values
        plot_cwt_power_sidebyside(idx_healthy, idx_unhealthy, coi_line_array, df_timedomain, graph_outputdir, h_ma, uh_ma, freqs, title=title, stepid=stepid, ntraces=ntraces)
    else:
        plot_cwt_power_sidebyside(idx_healthy, idx_unhealthy, coi_line_array, df_timedomain, graph_outputdir, h_m, uh_m, freqs, title=title, stepid=stepid, ntraces=ntraces)

    df_cwt = pd.DataFrame(features_flatten_sample_list, dtype=float)

    colums = [str(x) for x in range(df_cwt.shape[1])]
    colums[-1] = "label"
    df_cwt.columns = colums

    if enable_anscomb:
        print("computing anscombe...")
        df_cwt = get_anscombe(df_cwt)

    print(df_cwt)

    # plot_cwt_pca(df_cwt, title, graph_outputdir)

    # df_cwt_pca = pd.DataFrame(PCA(n_components=2).fit_transform(df_cwt.iloc[:, :-1].values))
    # df_cwt_pca["label"] = df_cwt["label"]

    return df_cwt, results_cwt_matrix_healthy, results_cwt_matrix_unhealthy


if __name__ == "__main__":
    print("args: output_dir dataset_filepath test_size")
    print("********************************************************************")
    # iris = datasets.load_iris()
    # X = iris.data[:, :100]
    # y = iris.target
    # dummy_run(X, y, 40, "dummy_iris.txt")
    # print("********************************************************************")
    # X, y = make_blobs(n_samples=50, centers=2, n_features=100, center_box=(0, 10))
    # dummy_run(X, y, 40, "dummy_blob.txt")
    # print("********************************************************************")
    # exit(0)

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        dataset_folder = sys.argv[2]
        n_splits = int(sys.argv[3])
        n_repeats = int(sys.argv[4])
        cwt_low_pass_filter = int(sys.argv[5])
        cwt_high_pass_filter = int(sys.argv[6])
        n_process = int(sys.argv[7])
    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_folder)
    print("n_splits=", n_splits)
    print("n_repeats=", n_repeats)
    print("n_repeats=", n_repeats)
    print("cwt_high_pass_filter=", cwt_high_pass_filter)
    print("cwt_low_pass_filter=", cwt_low_pass_filter)
    print("loading dataset...")
    enable_downsample_df =True

    # if os.path.exists(output_dir):
    #     print("purge %s..." % output_dir)
    #     try:
    #         shutil.rmtree(output_dir)
    #     except IOError:
    #         print("file not found.")

    files = glob2.glob(dataset_folder) #find datset files
    files = [file.replace("\\", '/') for file in files]
    # files = [file.replace("\\", '/') for file in files if 'activity' in file]
    print("found %d files." % len(files))

    MULTI_THREADING_ENABLED = (n_process > 0)
    print("MULTI_THREADING_ENABLED=", MULTI_THREADING_ENABLED)

    # if MULTI_THREADING_ENABLED:
    #     pool = Pool(processes=n_process)
    #     for file in files:
    #         data_frame_original, data_frame_timed_no_norm, data_frame_timed_norm, data_frame_timed_norm_anscombe, \
    #         data_frame_cwt_no_norm, data_frame_median_norm_cwt, data_frame_median_norm_cwt_anscombe = load_df_from_datasets(enable_downsample_df, output_dir, file, hi_pass_filter=cwt_high_pass_filter)
    #         thresh_i, thresh_z, days, farm_id, option, sampling = parse_param_from_filename(file)
    #
    #         print("thresh_i=", thresh_i)
    #         print("thresh_z=", thresh_z)
    #         print("days=", days)
    #         print("farm_id=", farm_id)
    #         print("option=", option)
    #         pool.apply_async(process_data_frame,
    #                          (output_dir, data_frame_median_norm_cwt_anscombe, thresh_i, thresh_z, days, farm_id, "norm_cwt_anscombe", n_splits, n_repeats,
    #                            sampling, enable_downsample_df,))
    #         pool.apply_async(process_data_frame,
    #                          (output_dir, data_frame_median_norm_cwt, thresh_i, thresh_z, days, farm_id, "norm_cwt", n_splits, n_repeats,
    #                            sampling, enable_downsample_df,))
    #
    #     pool.close()
    #     pool.join()
    #     pool.terminate()
    # else:
    for file in files:
        data_frame_original, data_frame_timed_no_norm, data_frame_timed_norm, data_frame_timed_norm_anscombe, \
        data_frame_cwt_no_norm, data_frame_median_norm_cwt, data_frame_median_norm_cwt_anscombe = load_df_from_datasets(enable_downsample_df, output_dir, file, hi_pass_filter=cwt_high_pass_filter, n_process=n_process)
        thresh_i, thresh_z, days, farm_id, option, sampling = parse_param_from_filename(file)

        # data_frame_original, data_frame, data_frame_cwt = load_matlab_dataset(file)
        print("thresh_i=", thresh_i)
        print("thresh_z=", thresh_z)
        print("days=", days)
        print("farm_id=", farm_id)
        print("option=", option)

        process_data_frame(output_dir, data_frame_median_norm_cwt_anscombe, thresh_i, thresh_z, days, farm_id, "norm_cwt_anscombe", n_splits, n_repeats,
                           sampling, enable_downsample_df)

        process_data_frame(output_dir, data_frame_median_norm_cwt, thresh_i, thresh_z, days, farm_id, "norm_cwt", n_splits, n_repeats,
                           sampling, enable_downsample_df)

        process_data_frame(output_dir, data_frame_cwt_no_norm, thresh_i, thresh_z, days, farm_id, "cwt_no_norm", n_splits, n_repeats,
                           sampling, enable_downsample_df)

        process_data_frame(output_dir, data_frame_timed_no_norm, thresh_i, thresh_z, days, farm_id, "activity_no_norm", n_splits, n_repeats,
                           sampling, enable_downsample_df)

        process_data_frame(output_dir, data_frame_timed_norm, thresh_i, thresh_z, days, farm_id, "activity_median_norm", n_splits, n_repeats,
                           sampling, enable_downsample_df)

        process_data_frame(output_dir, data_frame_timed_norm_anscombe, thresh_i, thresh_z, days, farm_id, "activity_norm_anscombe", n_splits, n_repeats,
                           sampling, enable_downsample_df)

    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)

    files = [output_dir + "/" + file for file in os.listdir(output_dir) if file.endswith(".csv")]
    print("found %d files." % len(files))
    print("compiling final file...")
    df_final = pd.DataFrame()
    dfs = [pd.read_csv(file, sep=",") for file in files]
    df_final = pd.concat(dfs)
    filename = "%s/final_classification_report_cv_%d_%d.csv" % (output_dir, n_splits, n_repeats)
    df_final.to_csv(filename, sep=',', index=False)
    print(df_final)
    print("done")

    # exit(-1)
    # dataset_filepath1 = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\csv_db\\cedara_70091100056_720\\7_1\\training_sets\\cwt_.data"
    # dataset_filepath2 = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\csv_db\\delmas_70101200027_720\\7_1\\training_sets\\cwt_.data"
    #
    # _, data_frame1, _ = load_df_from_datasets(dataset_filepath1)
    # _, data_frame2, _ = load_df_from_datasets(dataset_filepath2)
    # process_cross_farm(data_frame1, data_frame2)
