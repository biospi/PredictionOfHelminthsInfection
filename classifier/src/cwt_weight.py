import gc
import os
import pathlib
from datetime import datetime
from sys import exit

# sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet
import pywt
import scikitplot as skplt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions
from scipy import interp
from scipy.interpolate import UnivariateSpline
from scipy.signal import chirp
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.svm import SVC
from sklearn.utils import shuffle
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (15, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

DATA_ = []

def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
        # i[i > 150] = -1
        s = pd.Series(i)
        s = s.interpolate(method="linear", limit_direction="both")
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
    y = chirp(t, f0, period, f1, method="logarithmic")
    plt.plot(t, y)
    plt.grid(alpha=0.25)
    plt.xlabel("t (seconds)")
    plt.show()
    return t, y


def low_pass_filter(signal, thresh=0.35, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal[:-1]


def compute_cwt(activity, hd=False):
    return compute_cwt_hd(activity)


def compute_cwt_sd(activity, scale=80):
    w = pywt.ContinuousWavelet("mexh")
    scales = even_list(scale)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)

    coefs, freqs = pywt.cwt(
        np.asarray(activity_i), scales, w, sampling_period=sampling_period
    )

    # print('shapes:')
    # print(coefs.shape)
    diff = coefs.shape[1]
    n = int(coefs.shape[1] / 10)
    coefs = coefs[:, n:-n]
    diff = diff - coefs.shape[1]
    # print(coefs.shape, diff)

    cwt = [element for tupl in coefs for element in tupl]
    # indexes = np.asarray(list(range(coef.shape[1])))
    indexes = []
    # cwt = [x if x > -0.1 else 0 for x in cwt]
    return cwt, coefs, freqs, indexes, scales, 1, "morlet", []


def mask_cwt(cwt, coi):
    print("masking cwt...")
    for i in range(coi.shape[0]):
        col = cwt[:, i]
        max_index = int(coi[i])
        indexes_to_keep = np.array(list(range(max_index, col.shape[0])))
        total_indexes = np.array(range(col.shape[0]))
        diff = list(set(indexes_to_keep).symmetric_difference(total_indexes))
        # print(indexes_to_keep)
        if len(indexes_to_keep) == 0:
            continue
        col[indexes_to_keep] = -1
    return cwt


def compute_cwt_hd(activity, scale=10):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    scales = even_list(scale)
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    delta_t = (x[1] - x[0]) * 1
    wavelet_type = "morlet"
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        y, delta_t, wavelet=wavelet_type
    )
    cwt = coefs.flatten()
    indexes = []
    return cwt.real, coefs.real, freqs, indexes, scales, delta_t, wavelet_type, coi


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


def process_df(df, data):
    (
        scales,
        delta_t,
        wavelet_type,
        class0_mean,
        coefs_class0_mean,
        class1_mean,
        coefs_class1_mean,
        coefs_herd_mean,
        herd_mean,
    ) = data

    X = df[df.columns[0 : df.shape[1] - 1]]
    y = df["target"]
    dfs = []
    cwt_coefs_data = []
    df_x = pd.DataFrame(X.values[:, :])
    df_x["target"] = np.array(y)
    print("window:")
    print(df_x)
    dfs.append(df_x)
    cwt_coefs_data.append(
        (
            scales,
            delta_t,
            wavelet_type,
            class0_mean,
            class1_mean,
            herd_mean,
            coefs_class0_mean,
            coefs_class1_mean,
            coefs_herd_mean[:, :],
        )
    )

    return dfs, cwt_coefs_data


def chunck_df(days, df, data, w_day_step=None):
    (
        scales,
        delta_t,
        wavelet_type,
        class0_mean,
        coefs_class0_mean,
        class1_mean,
        coefs_class1_mean,
        coefs_herd_mean,
        herd_mean,
    ) = data

    X = df[df.columns[0 : df.shape[1] - 1]]
    y = df["target"]

    n_week = int(days / 7)
    chunch_size = int((X.shape[1] / n_week) / 1)
    step = int((X.shape[1] / (n_week * 7)) * w_day_step)

    print(
        "step size is %d, chunch_size is %d, n_week is %d" % (step, chunch_size, n_week)
    )
    dfs = []
    cwt_coefs_data = []

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
        df_x["label"] = np.array(y)
        print("window:")
        print(df_x)
        dfs.append(df_x)

        fig, axs = plt.subplots(2, 1, facecolor="white")
        axs[0].pcolormesh(coefs_class0_mean[start:end], cmap="viridis")
        axs[0].set_yscale("log")
        axs[1].pcolormesh(coefs_class1_mean[start:end], cmap="viridis")
        axs[1].set_yscale("log")
        fig.show()
        plt.close(fig)
        plt.close()
        fig.clear()

        cwt_coefs_data.append(
            (
                scales,
                delta_t,
                wavelet_type,
                class0_mean[start:end],
                class1_mean[start:end],
                herd_mean[start:end],
                coefs_class0_mean[start:end],
                coefs_class1_mean[start:end],
                coefs_herd_mean[:, start:end],
            )
        )

    return dfs, cwt_coefs_data


def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    # if output_dim not in [1, 2, 3]:
    #     raise ValueError("available dimension for features reduction are 1, 2 and 3.")
    # if output_dim == 3:
    #     X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])))
    #     y_train = np.append(y_train, (3, 4))
    #     X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1]), np.ones(X_train.shape[1])])))
    #     y_test = np.append(y_test, (3, 4))
    # if output_dim == 2:
    #     X_train = np.vstack((X_train, np.array([np.zeros(X_train.shape[1])])))
    #     y_train = np.append(y_train, 3)
    #     X_test = np.vstack((X_test, np.array([np.zeros(X_test.shape[1])])))
    #     y_test = np.append(y_test, 3)
    clf = LDA(n_components=output_dim)
    X_train = clf.fit_transform(X_train, y_train)[0]
    X_test = clf.fit_transform(X_test, y_test)[0]
    # if output_dim != 1:
    #     X_train = X_train[0:-(output_dim - 1)]
    #     y_train = y_train[0:-(output_dim - 1)]
    #     X_test = X_test[0:-(output_dim - 1)]
    #     y_test = y_test[0:-(output_dim - 1)]

    return X_train, X_test, y_train, y_test, clf


def process_fold(n, X, y, i, dim_reduc=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=i, stratify=y
    )
    print(X_train.shape, X_test.shape, y)

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == "LDA":
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_lda(
            n, X_train, X_test, y_train, y_test
        )

    print(X_train_reduced.shape, X_test_reduced.shape, y)
    X_reduced = np.concatenate((X_train_reduced, X_test_reduced), axis=0)
    print(y_train_reduced.shape, y_test_reduced.shape)
    y_reduced = np.concatenate((y_train_reduced, y_test_reduced), axis=0)

    return (
        X_reduced,
        y_reduced,
        X_train_reduced,
        X_test_reduced,
        y_train_reduced,
        y_test_reduced,
    )


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
    precision_recall_fscore_support_result = precision_recall_fscore_support(
        test_y, pred_y, average=None, labels=[0, 1]
    )
    precision_false = precision_recall_fscore_support_result[0][0]
    precision_true = precision_recall_fscore_support_result[0][1]
    recall_false = precision_recall_fscore_support_result[1][0]
    recall_true = precision_recall_fscore_support_result[1][1]
    fscore_false = precision_recall_fscore_support_result[2][0]
    fscore_true = precision_recall_fscore_support_result[2][1]
    support_false = precision_recall_fscore_support_result[3][0]
    support_true = precision_recall_fscore_support_result[3][1]
    return (
        precision_false,
        precision_true,
        recall_false,
        recall_true,
        fscore_false,
        fscore_true,
        support_false,
        support_true,
    )


def plot_2D_decision_boundaries(
    X_lda,
    y_lda,
    X_test,
    y_test,
    title,
    clf,
    filename="",
    days=None,
    resolution=None,
    folder=None,
    i=0,
    df_id=0,
    sub_dir_name=None,
    n_bin=8,
    save=True,
):
    print("graph...")
    # plt.subplots_adjust(top=0.75)
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # plt.subplots_adjust(top=0.75)
    min = abs(X_lda.min()) + 1
    max = abs(X_lda.max()) + 1
    print(X_lda.shape)
    print(min, max)
    if np.max([min, max]) > 100:
        return
    xx, yy = np.mgrid[-min:max:0.01, -min:max:0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    offset_r = 0
    offset_g = 0
    offset_b = 0
    colors = [
        ((77 + offset_r) / 255, (157 + offset_g) / 255, (210 + offset_b) / 255),
        (1, 1, 1),
        ((255 + offset_r) / 255, (177 + offset_g) / 255, (106 + offset_b) / 255),
    ]
    cm = LinearSegmentedColormap.from_list("name", colors, N=n_bin)

    for _ in range(0, 1):
        contour = ax.contourf(
            xx,
            yy,
            probs,
            n_bin,
            cmap=cm,
            antialiased=False,
            vmin=0,
            vmax=1,
            alpha=0.3,
            linewidth=0,
            linestyles="dashed",
            zorder=-1,
        )
        ax.contour(
            contour, cmap=cm, linewidth=1, linestyles="dashed", zorder=-1, alpha=1
        )

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
    ax.scatter(
        X_lda_0[:, 0],
        X_lda_0[:, 1],
        c=(39 / 255, 111 / 255, 158 / 255),
        s=marker_size,
        vmin=-0.2,
        vmax=1.2,
        edgecolor=(49 / 255, 121 / 255, 168 / 255),
        linewidth=0,
        marker="s",
        alpha=0.7,
        label="Class0 (Healthy)",
        zorder=1,
    )

    ax.scatter(
        X_lda_1[:, 0],
        X_lda_1[:, 1],
        c=(251 / 255, 119 / 255, 0 / 255),
        s=marker_size,
        vmin=-0.2,
        vmax=1.2,
        edgecolor=(255 / 255, 129 / 255, 10 / 255),
        linewidth=0,
        marker="^",
        alpha=0.7,
        label="Class1 (Unhealthy)",
        zorder=1,
    )

    ax.scatter(
        X_lda_0_t[:, 0],
        X_lda_0_t[:, 1],
        s=marker_size - 10,
        vmin=-0.2,
        vmax=1.2,
        edgecolor="black",
        facecolors="none",
        label="Test data",
        zorder=1,
    )

    ax.scatter(
        X_lda_1_t[:, 0],
        X_lda_1_t[:, 1],
        s=marker_size - 10,
        vmin=-0.2,
        vmax=1.2,
        edgecolor="black",
        facecolors="none",
        zorder=1,
    )

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    ax.contour(
        xx, yy, probs, levels=[0.5], cmap="Reds", vmin=0, vmax=0.6, linewidth=0.1
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    handles, labels = ax.get_legend_handles_labels()
    db_line = Line2D(
        [0], [0], color=(183 / 255, 37 / 255, 42 / 255), label="Decision boundary"
    )
    handles.append(db_line)

    plt.legend(loc=4, fancybox=True, framealpha=0.4, handles=handles)
    plt.title(title)
    ttl = ax.title
    ttl.set_position([0.57, 0.97])
    # plt.tight_layout()

    # path = filename + '\\' + str(resolution) + '\\'
    # path_file = path + "%d_p.png" % days
    # pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(path_file, bbox_inches='tight')
    if save:
        path = "%s/%s/decision_boundaries_graphs/df%d/" % (folder, sub_dir_name, df_id)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "iter_%d.png" % (i)
        final_path = "%s/%s" % (path, filename)
        print(final_path)
        try:
            plt.savefig(final_path, bbox_inches="tight")
        except FileNotFoundError as e:
            print(e)
            exit()

        plt.close()
        # fig.show()
        plt.close()
        fig.clear()
    else:
        fig.show()


def plot_2D_decision_boundaries_(
    X, y, X_test, title, clf, folder=None, i=0, df_id=None, sub_dir_name=None, save=True
):

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
    scatter_kwargs = {"s": 120, "edgecolor": None, "alpha": 0.7}
    contourf_kwargs = {"alpha": 0.2}
    scatter_highlight_kwargs = {"s": 120, "label": "Test data", "alpha": 0.7}
    plot_decision_regions(
        X,
        y,
        clf=clf,
        legend=2,
        X_highlight=X_test,
        scatter_kwargs=scatter_kwargs,
        contourf_kwargs=contourf_kwargs,
        scatter_highlight_kwargs=scatter_highlight_kwargs,
    )
    plt.title(title)
    if save:
        path = "%s/%s/decision_boundaries_graphs/df%d/" % (folder, sub_dir_name, df_id)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "iter_%d.png" % (i)
        final_path = "%s/%s" % (path, filename)
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


def compute_model_loo(
    X,
    y,
    X_train,
    y_train,
    X_test,
    y_test,
    farm_id,
    n,
    clf=None,
    dim_reduc_name="LDA",
    resolution="10min",
    df_id=None,
    days=None,
):
    # X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(2, X, y, n, dim_reduc=dim_reduc_name)
    print("fitting...")
    clf.fit(X_train, y_train)
    # clf = clf.best_estimator_
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    (
        precision_false,
        precision_true,
        recall_false,
        recall_true,
        fscore_false,
        fscore_true,
        support_false,
        support_true,
    ) = get_prec_recall_fscore_support(y_test, y_pred)

    if np.isnan(recall_false):
        recall_false = -1
    if np.isnan(recall_true):
        recall_true = -1
    if np.isnan(p_y_false):
        p_y_false = -1
    if np.isnan(p_y_true):
        p_y_true = -1

    print(
        (
            "LREG",
            "" if dim_reduc_name is None else dim_reduc_name,
            2,
            3,
            0,
            acc * 100,
            precision_false * 100,
            precision_true * 100,
            recall_false * 100,
            recall_true * 100,
            p_y_false * 100,
            p_y_true * 100,
            np.count_nonzero(y == 0),
            np.count_nonzero(y == 1),
            np.count_nonzero(y == 0),
            np.count_nonzero(y == 1),
            np.count_nonzero(y_test == 0),
            np.count_nonzero(y_test == 1),
            resolution,
        )
    )

    title = "%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;" "class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s\n" % (
        "LREG",
        "" if dim_reduc_name is None else dim_reduc_name,
        2,
        3,
        0,
        acc * 100,
        precision_false * 100,
        precision_true * 100,
        recall_false * 100,
        recall_true * 100,
        p_y_false * 100,
        p_y_true * 100,
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
        np.count_nonzero(y_test == 0),
        np.count_nonzero(y_test == 1),
        resolution,
    )

    sub_dir_name = "days_%d_class0_%d_class1_%d" % (
        days,
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
    )

    plot_2D_decision_boundaries(
        X,
        y,
        X_test,
        y_test,
        title,
        clf,
        folder="%s\\%d\\transition\\classifier_transit" % (farm_id, days),
        i=n,
        df_id=df_id,
        sub_dir_name=sub_dir_name,
    )
    return (
        acc,
        precision_false,
        precision_true,
        recall_false,
        recall_true,
        fscore_false,
        fscore_true,
        support_false,
        support_true,
        sub_dir_name,
    )


def compute_model(
    X,
    y,
    n,
    farm_id,
    clf=None,
    dim_reduc_name="LDA",
    resolution="10min",
    df_id=None,
    days=None,
):
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
    (
        precision_false,
        precision_true,
        recall_false,
        recall_true,
        fscore_false,
        fscore_true,
        support_false,
        support_true,
    ) = get_prec_recall_fscore_support(y_test, y_pred)

    if np.isnan(recall_false):
        recall_false = -1
    if np.isnan(recall_true):
        recall_true = -1
    if np.isnan(p_y_false):
        p_y_false = -1
    if np.isnan(p_y_true):
        p_y_true = -1

    print(
        (
            "LREG",
            "" if dim_reduc_name is None else dim_reduc_name,
            2,
            3,
            0,
            acc * 100,
            precision_false * 100,
            precision_true * 100,
            recall_false * 100,
            recall_true * 100,
            p_y_false * 100,
            p_y_true * 100,
            np.count_nonzero(y == 0),
            np.count_nonzero(y == 1),
            np.count_nonzero(y == 0),
            np.count_nonzero(y == 1),
            np.count_nonzero(y_test == 0),
            np.count_nonzero(y_test == 1),
            resolution,
        )
    )

    title = "%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;" "class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s\n" % (
        "LREG",
        "" if dim_reduc_name is None else dim_reduc_name,
        2,
        3,
        0,
        acc * 100,
        precision_false * 100,
        precision_true * 100,
        recall_false * 100,
        recall_true * 100,
        p_y_false * 100,
        p_y_true * 100,
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
        np.count_nonzero(y_test == 0),
        np.count_nonzero(y_test == 1),
        resolution,
    )

    sub_dir_name = "days_%d_class0_%d_class1_%d" % (
        days,
        np.count_nonzero(y == 0),
        np.count_nonzero(y == 1),
    )

    plot_2D_decision_boundaries(
        X,
        y,
        X_test,
        y_test,
        title,
        clf,
        folder="%s\\%d\\transition\\classifier_transit" % (farm_id, days),
        i=n,
        df_id=df_id,
        sub_dir_name=sub_dir_name,
    )

    print(n, acc)
    return (
        acc,
        precision_false,
        precision_true,
        recall_false,
        recall_true,
        fscore_false,
        fscore_true,
        support_false,
        support_true,
        sub_dir_name,
    )


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


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, i, fig, prec_data_str):
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="tab:blue",
        label=r"Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )"
        % (mean_auc, lo, hi),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    confidence_lower, confidence_upper = get_conf_interval(tprs, mean_fpr)

    ax.fill_between(
        mean_fpr, confidence_lower, confidence_upper, color="tab:blue", alpha=0.2
    )
    # label=r'$\pm$ 1 std. dev.')

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        # title="Receiver operating characteristic iteration %d" % (i + 1)
    )
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/" % (out_dir)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = "%s/%s" % (path, "roc_%d_%s.png" % (i, prec_data_str))
    final_path = final_path.replace("/", "'").replace("'", "\\").replace("\\", "/")
    print(final_path)
    fig.savefig(final_path)


def process_transit(dfs, days, resolution, farm_id):
    data_acc, data_pf, data_pt, data_rf, data_rt, data_ff, data_ft, data_sf, data_st = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
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

        clf = SVC(kernel="linear", probability=True)
        X, y = process_data_frame_(data_frame)
        X, _, y, _, _ = reduce_lda(2, X, X, y, y)
        # model.fit(X, y)
        (
            acc_list,
            p_f_list,
            p_t_list,
            recall_f_list,
            recall_t_list,
            fscore_f_list,
            fscore_t_list,
            support_f_list,
            support_t_list,
        ) = ([], [], [], [], [], [], [], [], [])
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        # loo = LeaveOneOut()
        # loo.get_n_splits(X, y)
        rkf = RepeatedKFold(
            n_splits=10,
            n_repeats=10,
            random_state=int((datetime.now().microsecond) / 10),
        )

        for n, (train_index, test_index) in enumerate(rkf.split(X)):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            (
                acc,
                precision_false,
                precision_true,
                recall_false,
                recall_true,
                fscore_false,
                fscore_true,
                support_false,
                support_true,
                sub_dir_name,
            ) = compute_model_loo(
                X,
                y,
                X_train,
                y_train,
                X_test,
                y_test,
                farm_id,
                n,
                clf=clf,
                df_id=id,
                days=days,
            )

            # for n, clf in enumerate(model.estimators_):
            #     try:
            #         acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name = compute_model(
            #             X, y, n, farm_id, clf=clf, df_id=id, days=days)
            #     except ValueError as e:
            #         print(e)
            #         continue

            viz = plot_roc_curve(
                clf, X, y, name="", label="_Hidden", alpha=0, lw=1, ax=ax
            )
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
        print("precisions...")
        prec_data_str = "%.2f_%.2f" % (np.mean(p_f_list), np.mean(p_t_list))
        print(id, prec_data_str)
        plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, id, fig, prec_data_str)
        fig.clear()

    ribbon_plot_dir = "%s\\%d\\transition\\ribbon_transit\\%s" % (
        farm_id,
        days,
        sub_dir_name,
    )
    pathlib.Path(ribbon_plot_dir).mkdir(parents=True, exist_ok=True)

    plot_(
        ribbon_plot_dir,
        data_acc,
        "Classifier accuracy over time during increase of the FAMACHA score",
        "model accuracy in %",
        days,
    )


def plot_(path, data, title, y_label, days):
    df = pd.DataFrame.from_dict(data, orient="index")
    print(df)
    time = []
    acc = []
    for index, row in df.iterrows():
        print(row[0], row[1])
        for n in range(df.shape[1]):
            time.append(index)
            acc.append(row[n])
    data_dict = {"time": time, "acc": acc}
    df = pd.DataFrame.from_dict(data_dict)
    print(df)
    time_axis = interpolate_time(np.arange(days + 1), len(df["time"]))
    time_axis = time_axis.tolist()
    time_axis_s = []
    for t in time_axis:
        time_axis_s.append("%d" % t)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x=df["time"], y="acc", data=df, marker="o", ax=ax)
    ax.set_title(title)
    # ax = df.copy().plot.box(grid=True, patch_artist=True, title=title, figsize=(10, 7))
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    m_d = max(df["time"].to_list()) + 1
    labels_ = interpolate_time(np.arange(15), m_d)
    l = []
    for i, item in enumerate(labels_):
        l.append("%.1f" % float(item))

    # labels = ['0'] + labels + ['0']
    # ax.set_xticklabels(labels)
    ticks = list(range(m_d))
    ax.set_xticks(ticks)
    ax.set_xticklabels(l)

    print("labels", labels)

    # ax.set_xticklabels(time_axis_s)
    file_path = "%s\\%s.png" % (path, y_label)
    plt.savefig(file_path)
    plt.show()


def plot(ax, data, title, y_label):
    df = pd.DataFrame.from_dict(data, orient="index")
    print(df)
    time = []
    acc = []
    for index, row in df.iterrows():
        print(row[0], row[1])
        for n in range(df.shape[1]):
            time.append(index)
            acc.append(row[n])
    data_dict = {"time": time, "acc": acc}
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


def process_data_frame_(data_frame, y_col="label"):
    data_frame = data_frame.fillna(-1)
    # cwt_shape = data_frame[data_frame.columns[0:2]].values
    X = data_frame[data_frame.columns[2 : data_frame.shape[1] - 1]].values
    print(X)
    X = normalize(X)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    return X, y


def get_cwt_data_frame(data_frame):
    global DATA_
    data_frame["target"] = (data_frame["target"].values == 1).astype(int)
    data_frame = data_frame.tail(4)
    DATA_ = []
    X = data_frame[data_frame.columns[0 : data_frame.shape[1] - 1]].values
    H = []
    for _, activity in enumerate(X):
        activity = np.asarray(activity)
        H.append(activity)
    herd_mean = np.average(H, axis=0)

    print("herd window:")
    print(pd.DataFrame(herd_mean).transpose())
    print("finished computing herd mean.")
    print("computing herd cwt")
    cwt_herd, coefs_herd_mean, freqs_h, _, _, _, _, coi = compute_cwt(herd_mean)
    DATA_.append({"coef_shape": coefs_herd_mean.shape, "freqs": freqs_h})
    print("finished calculating herd cwt.")

    X_cwt = pd.DataFrame()
    cpt = 0
    class0 = []
    class1 = []
    class0_t = []
    class1_t = []
    for activity, (i, row) in zip(X, data_frame.iterrows()):
        # activity = interpolate(activity)
        activity = np.asarray(activity)
        activity_o = activity.copy()
        # activity = np.divide(activity, herd_mean)
        print(len(activity), "%d/%d ..." % (cpt, len(X)))
        cwt, coefs, freqs, indexes, scales, delta_t, wavelet_type, coi = compute_cwt(
            activity
        )
        print(len(activity), len(cwt))
        X_cwt = X_cwt.append(dict(enumerate(np.array(cwt))), ignore_index=True)
        cpt += 1
        target = data_frame.at[i, "target"]
        if target == 0:
            class0.append(cwt)
            class0_t.append(activity_o)
        if target == 1:
            class1.append(cwt)
            class1_t.append(activity_o)

    class0_mean = np.average(class0_t, axis=0)
    _, coefs_class0_mean, _, _, _, _, _, coi = compute_cwt(class0_mean)
    class1_mean = np.average(class1_t, axis=0)
    _, coefs_class1_mean, _, _, _, _, _, coi = compute_cwt(class1_mean)

    y = data_frame["target"].values.flatten()
    y = y.astype(int)
    X_cwt["target"] = y

    print(data_frame)
    print(X_cwt)

    return (
        X_cwt,
        len(class0),
        len(class1),
        (
            scales,
            delta_t,
            wavelet_type,
            class0_mean,
            coefs_class0_mean,
            class1_mean,
            coefs_class1_mean,
            coefs_herd_mean,
            herd_mean,
        ),
    )


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(1, 1 + 2 * top_features),
        feature_names[top_coefficients],
        rotation=60,
        ha="right",
    )
    plt.show()


def pad(A, length):
    arr = np.zeros(length)
    arr[: len(A)] = A
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


def plot_cwt_coefs(
    fig,
    axs,
    x,
    y,
    coefs_class0_mean,
    i=0,
    vmin_map=None,
    vmax_map=None,
    auto_scale=True,
    title="title",
):
    # fig, ax = plt.subplots(figsize=(9, 4.8))
    ax = axs[x, y]
    ax.grid(False)

    if auto_scale or i <= 3:
        vmin_map = coefs_class0_mean.min()
        vmax_map = coefs_class0_mean.max()

    # time_axis = interpolate_time(np.arange(days+1), x_axis_lenght)
    time_axis = list(range(coefs_class0_mean.shape[1]))
    im = ax.pcolormesh(
        time_axis,
        DATA_[0]["freqs"],
        coefs_class0_mean,
        cmap="viridis",
        vmin=vmin_map,
        vmax=vmax_map,
    )
    fig.colorbar(im, ax=ax)
    ax.set_yscale("log")
    ax.title.set_text(title)
    ax.set(xlabel="$Time (days)$", ylabel="$Frequency$")


def pot_icwt(
    axs,
    x,
    y,
    iwave0,
    ymin2,
    ymax2,
    ymin2_,
    ymax2_,
    days=0,
    i=0,
    auto_scale=False,
    title="title",
):
    if i in [0, 1, 2]:
        ymin2 = ymin2_
        ymax2 = ymax2_
    if auto_scale:
        ymin2 = min(iwave0)
        ymax2 = max(iwave0)
    try:
        with plt.style.context("seaborn-white"):
            print("pot_icwt...")
            ax = axs[x, y]
            ax.grid(False)
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            # fig, ax = plt.subplots(figsize=(25, 4.8))
            time_axis = interpolate_time(np.arange(days + 1), len(iwave0))
            ax.plot(time_axis, iwave0)
            # y_mean = [np.mean(iwave0)] * len(time_axis)
            # ax.plot(time_axis, y_mean, label='Mean', linestyle='--')
            del iwave0
            print([ymin2, ymax2])
            ax.set_ylim([ymin2, ymax2])
            ax.title.set_text(title)
            ax.set(xlabel="$Time (days)$", ylabel="Activity")
    except ValueError as e:
        print(e)


def save_roc_curve(y_test, y_probas, title, options, folder, i=0, j=0):
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    # plt.title('ROC Curves %s' % title)
    split = title.split("\n")
    title = "ROC Curves"
    skplt.metrics.plot_roc(y_test, y_probas, title=title, title_fontsize="medium")
    path = "%s/roc_curve/" % folder
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    final_path = "%s/%s" % (path, "roc_%d_%d.png" % (j, i))
    final_path = final_path.replace("/", "'").replace("'", "\\").replace("\\", "/")
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

    return proba_0, proba_1


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
    selector = RFECV(
        clf, step=1, cv=5, n_jobs=n_job, min_features_to_select=n_features_to_select
    )
    selector = selector.fit(X, y)
    ranking = selector.ranking_
    # ranking = np.reshape(ranking, shape)
    ranking[ranking != 1] = 0
    # ranking = minmax_scale(ranking, feature_range=(0, 1))
    print("rec_feature_selection done")
    # returns map with location of important features 1 for important 0 for not important
    return ranking


def get_eli5_weight(aux1, i):
    class0 = aux1[aux1.target == aux1.target.unique()[i]]
    class0 = class0[class0.feature != "<BIAS>"]
    class0["feature"] = class0["feature"].str.replace("x", "")
    class0["feature"] = class0["feature"].apply(int)
    class0 = class0.sort_values("feature")
    weight0 = class0["weight"].values
    return weight0


def get_weight_map_data(
    weight_array, shape, input, scales, delta_t, wavelet_type, force_abs=False
):
    # input = minmax_scale(input, feature_range=(-1, 1))
    # weight_array = minmax_scale(weight_array, feature_range=(0, 1))
    # input = None
    weight_matrix = np.reshape(weight_array, shape)
    if force_abs:
        weight_matrix = np.abs(weight_matrix)
    if input is not None:
        weighted_input = np.multiply(weight_matrix, input)
    else:
        weighted_input = weight_matrix

    # n = int(weighted_input.shape[1] / 10)
    # weighted_input = weighted_input[:, n:weighted_input.shape[0]-n]

    iwave = wavelet.icwt(weighted_input, scales, delta_t, wavelet=wavelet_type)
    iwave = np.real(iwave)
    if weighted_input[weighted_input == 0].size > 0:
        weighted_input[weighted_input == 0] = np.nan
    return weighted_input, iwave


def get_top_weight(weight, scale=2):
    n_to_keep = int(weight.shape[0] / scale)  # keep half of best features
    print("n_to_keep=%d/%d" % (n_to_keep, weight.shape[0]))
    top_features_indexes = weight.argsort()[-n_to_keep:][::-1]
    mask = np.zeros(weight.shape[0])
    mask[top_features_indexes] = weight[top_features_indexes]
    return mask


def get_min_max(data, ignore=[0, 1, 2, 5, 6, 7, 8]):
    cwt_list = []
    icwt_list = []
    for n, item in enumerate(data):
        if n in ignore:  # todo fix exclude certain graphs axis
            continue

        a = item[0][~np.isnan(item[0])]
        b = item[1][~np.isnan(item[1])]
        cwt_list.append(a.min())
        cwt_list.append(a.max())
        icwt_list.append(b.min())
        icwt_list.append(b.max())

    return min(cwt_list), max(cwt_list), min(icwt_list), max(icwt_list)


def explain_cwt(days, dfs, data, out_dir, class0_count, class1_count):
    global DATA_ #todo clean up this mess
    plt.clf()
    print("process...", days)

    for i, df in enumerate(dfs):
        (
            scales,
            delta_t,
            wavelet_type,
            class0_mean,
            class1_mean,
            herd_mean,
            coefs_class0_mean,
            coefs_class1_mean,
            coefs_herd_mean,
        ) = data[i]
        # df = shuffle(df)
        X = df[df.columns[0 : df.shape[1] - 1]]
        X = X.fillna(-1)
        y = df["target"].values

        X_train, X_test, y_train, y_test = X, X, y, y

        clf = SVC(kernel="linear", probability=True)

        print("fit...")
        X_train_svm = X_train.copy()
        y_train_svm = y_train.copy()

        clf.fit(X_train_svm, y_train_svm)

        print("explain_prediction...")

        weight0 = clf.coef_[0]

        pad_value = abs(np.prod(DATA_[0]["coef_shape"]) - weight0.shape[0])
        for n in range(pad_value):
            weight0 = np.append(weight0, 0)

        weight0_best = get_top_weight(weight0)

        weight0_best_2 = get_top_weight(weight0, scale=20)

        print("building figure...")
        shape = DATA_[0]["coef_shape"]
        data_to_plot = []
        data_to_plot.append(
            (coefs_herd_mean, herd_mean, "Mean herd activity", "Mean cwt of herd")
        )
        data_to_plot.append(
            (
                coefs_class0_mean,
                class0_mean,
                f"Mean activity of Healthy animals ({class0_count})",
                "Mean cwt of Healthy animals"
            )
        )
        data_to_plot.append(
            (
                coefs_class1_mean,
                class1_mean,
                f"Mean activity of Unhealthy animals ({class1_count})",
                "Mean cwt of Unhealthy animals"
            )
        )

        data_to_plot.append(
            get_weight_map_data(
                weight0, shape, coefs_class0_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Healthy animals * features weight", "Inverse cwt")
        )
        data_to_plot.append(
            get_weight_map_data(
                weight0, shape, coefs_class1_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Unhealthy animals * features weight", "Inverse")
        )

        data_to_plot.append(
            get_weight_map_data(
                weight0_best, shape, coefs_class0_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Healthy animals * features weight (top 50% features)", "Inverse cwt")
        )
        data_to_plot.append(
            get_weight_map_data(
                weight0_best, shape, coefs_class1_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Unhealthy animals * features weight (top 50% features)", "Inverse")
        )
        data_to_plot.append(
            get_weight_map_data(
                weight0_best_2, shape, coefs_class0_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Healthy animals * features weight (top 20% features)", "Inverse cwt")
        )
        data_to_plot.append(
            get_weight_map_data(
                weight0_best_2, shape, coefs_class1_mean, scales, delta_t, wavelet_type
            )
            + ("Mean cwt of Unhealthy animals * features weight (top 20% features)", "Inverse")
        )

        with plt.style.context("seaborn-white"):
            fig, axs = plt.subplots(len(data_to_plot), 2, facecolor="white")
            fig.set_size_inches(17, 2.7 * len(data_to_plot))

            v_min_map, v_max_map, ymin2, ymax2 = get_min_max(data_to_plot)
            print("v_min_map, v_max_map", v_min_map, v_max_map)
            # v_min_map = -0.13794706803426268
            # v_max_map = 0.11997465366804147
            v_min_map_, v_max_map_, ymin2_, ymax2_ = get_min_max(
                data_to_plot, ignore=[3, 4, 5, 6, 7, 8]
            )

            for i, item in enumerate(data_to_plot):
                plot_cwt_coefs(
                    fig,
                    axs,
                    i,
                    0 if i > 2 else 1,
                    item[0],
                    i=i,
                    vmin_map=v_min_map,
                    vmax_map=v_max_map,
                    auto_scale=False,
                    title=item[2]
                )
                pot_icwt(
                    axs,
                    i,
                    1 if i > 2 else 0,
                    item[1],
                    ymin2,
                    ymax2,
                    ymin2_,
                    ymax2_,
                    days=days,
                    i=i,
                    auto_scale=False,
                    title=item[3]
                )
            fig.tight_layout()
            fig.show()
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = out_dir / "cwt_weight.png"
            fig.savefig(str(filename), dpi=100, facecolor="white")


def slice_df(df):
    print(df["famacha_score"].value_counts())
    print(df)
    df = df.loc[:, :"label"]
    np.random.seed(0)
    df = df.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.fillna(-1)
    df = shuffle(df)
    df["label"] = df["label"].map({True: 1, False: 0})
    print(df)
    return df


if __name__ == "__main__":
   print()