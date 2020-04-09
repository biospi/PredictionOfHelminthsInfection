import gc
import os
import shutil
from sys import exit

import eli5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet
import pywt
from scipy.signal import chirp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import normalize
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

# sns.set()
import matplotlib.pyplot as plt

DATA_ = []
CWT_RES = 1000000
TRAINING_DIR = "E:/Users/fo18103/PycharmProjects" \
               "/prediction_of_helminths_infection/training_data_generator_and_ml_classifier/src/"

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def interpolate(input_activity):
    try:
        i = np.array(input_activity, dtype=np.float)
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


def compute_cwt_(sig):
    print("computing cwt...")
    widths = np.arange(1, 31)
    coefs = signal.cwt(sig, signal.morlet, widths)
    # plt.imshow(coefs, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    #            vmax=abs(coefs).max(), vmin=-abs(coefs).max())
    # plt.show()
    # plt.matshow(coefs.real)
    # plt.show()
    cwt = [element for tupl in coefs.real for element in tupl]
    return cwt, coefs.real, widths, [], widths, 1, 'morlet'


def compute_cwt_(activity):
    w = pywt.ContinuousWavelet('morl')
    scales = even_list(40)
    sampling_frequency = 1 / 60
    sampling_period = 1 / sampling_frequency
    activity_i = interpolate(activity)
    coef, freqs = pywt.cwt(np.asarray(activity_i), scales, w, sampling_period=sampling_period)
    cwt = [element for tupl in coef for element in tupl]
    # indexes = np.asarray(list(range(coef.shape[1])))
    indexes = []
    return cwt, coef, freqs, indexes, scales, 1, 'morlet'


def compute_cwt(activity):
    print("compute_cwt...")
    # t, activity = dummy_sin()
    num_steps = len(activity)
    x = np.arange(num_steps)
    y = activity
    y = interpolate(y)

    delta_t = (x[1] - x[0])*2
    scales = np.arange(1, num_steps + 1) / 1
    freqs = 1 / (wavelet.Morlet().flambda() * scales)
    wavelet_type = 'morlet'
    # y = [0 if x is np.nan else x for x in y] #todo fix
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, delta_t, dj=CWT_RES, wavelet=wavelet_type, freqs=freqs)

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


META_DATA_LENGTH = 16
def find_type_for_mem_opt(df):
    data_col_n = df.iloc[[0]].size
    type_dict = {}
    for n, i in enumerate(range(0, data_col_n)):
        if n < (data_col_n - META_DATA_LENGTH):
            type_dict[str(i)] = np.float16
        else:
            type_dict[str(i)] = np.str
    del df
    type_dict[str(data_col_n-1)] = np.int
    type_dict[str(data_col_n-2)] = np.int
    type_dict[str(data_col_n-3)] = np.int
    type_dict[str(data_col_n-8)] = np.int
    type_dict[str(data_col_n - 9)] = np.int
    type_dict[str(data_col_n - 10)] = np.int
    type_dict[str(data_col_n - 11)] = np.int
    type_dict[str(data_col_n - 15)] = np.int
    return type_dict


def start(fname='', out_fname=None, fname_temp=None, fname_hum=None, resolution=None, days=None, f_config=None,
          farm_id=None, output_clf_transit=False, cwt1=None, cwt2=None, filter_delmas=None, filter_cedara=None):

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
    print(df)
    type_dict = find_type_for_mem_opt(df)
    df = pd.read_csv(fname, sep=",", header=None, dtype=type_dict)
    del type_dict

    # print(data_frame)
    sample_count = df.shape[1]
    hearder = [str(n) for n in range(0, sample_count)]
    hearder[-16] = "label"
    hearder[-15] = "elem_in_row"
    hearder[-14] = "date1"
    hearder[-13] = "date2"
    hearder[-12] = "serial"
    hearder[-11] = "famacha_score"
    hearder[-10] = "previous_famacha_score"
    hearder[-9] = "previous_famacha_score2"
    hearder[-8] = "previous_famacha_score3"

    hearder[-7] = "dtf1"
    hearder[-6] = "dtf2"
    hearder[-5] = "dtf3"
    hearder[-4] = "dtf4"

    hearder[-3] = "nd1"
    hearder[-2] = "nd2"
    hearder[-1] = "nd3"

    df.columns = hearder
    if filter_delmas:
        if days == 7 + 7 + 7 + 7:
            df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) & (df.previous_famacha_score3 == 1)) |
                    ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) & (df.previous_famacha_score3 == 1))]

        if days == 7 + 7 + 7:
            df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1)) |
                    ((df.famacha_score == 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1))]

        if days == 7 + 7:
            df = df[(df.nd1 == 7) & (df.nd2 == 7) & (df.nd3 == 7)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1)) |
                    ((df.famacha_score == 2) & (df.previous_famacha_score == 1))]

    if filter_cedara:
        if days == 7 + 7 + 7 + 7:
            df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd3 == 14)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) & (df.previous_famacha_score3 == 1)) |
                    ((df.famacha_score >= 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1) & (df.previous_famacha_score3 == 1))]

        if days == 7 + 7 + 7:
            df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd3 == 14)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1)) |
                    ((df.famacha_score >= 2) & (df.previous_famacha_score == 1) & (df.previous_famacha_score2 == 1))]

        if days == 7 + 7:
            df = df[(df.nd1 == 14) & (df.nd2 == 14) & (df.nd3 == 14)]
            df = df[((df.famacha_score == 1) & (df.previous_famacha_score == 1)) |
                    ((df.famacha_score >= 2) & (df.previous_famacha_score == 1))]

    df_0 = df
    print(df['famacha_score'].value_counts())
    print(df)
    df = df.loc[:, :'label']
    np.random.seed(0)
    df = df.sample(frac=1).reset_index(drop=True)
    # data_frame = data_frame.fillna(-1)
    df = shuffle(df)
    df['label'] = df['label'].map({True: 1, False: 0})
    print(df)
    if df.shape[0] == 0:
        return

    if output_clf_transit:
        print("output transit...")
        process_(df, days)
        return

    process(df, df_0, out_fname, df_temp=df_temp, df_hum=df_hum, resolution=resolution, farm_id=farm_id, days=days,
            f_config=f_config, out_dir="%s\\" % farm_id)


def chunck_df(df, chunk_size, labels, step=5000):
    dfs = []

    for i in range(0, int(df.shape[1]), step):
        start = i
        end = int(start + chunk_size-1)
        if end > int(df.shape[1]):
            end = int(df.shape[1]) - 1
        if abs(start - end) != chunk_size-1:
            continue
        print(start, end, abs(start - end))
        chunck = df.loc[:, str(start):str(end)]
        chunck["label"] = labels
        dfs.append(chunck)

    # for i in range(0, n_chunck):
    #     start = str(int(chunk_size*i))
    #     end = str(int(chunk_size * (i+1))-1)
    #     print(start, end)
    #     chunck = df.loc[:, start:end]
    #     chunck["label"] = labels
    #     dfs.append(chunck)
    return dfs


def reduce_lda(output_dim, X_train, X_test, y_train, y_test):
    # lda implementation require 3 input class for 2d output and 4 input class for 3d output
    if output_dim not in [1, 2, 3]:
        raise ValueError("available dimension for features reduction are 1, 2 and 3.")
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
    X_train = LDA(n_components=output_dim).fit_transform(X_train, y_train)
    X_test = LDA(n_components=output_dim).fit_transform(X_test, y_test)
    if output_dim != 1:
        X_train = X_train[0:-(output_dim - 1)]
        y_train = y_train[0:-(output_dim - 1)]
        X_test = X_test[0:-(output_dim - 1)]
        y_test = y_test[0:-(output_dim - 1)]

    return X_train, X_test, y_train, y_test


def process_fold(n, X, y, i, dim_reduc=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i, stratify=y)
    print(X_train.shape, X_test.shape, y)

    if dim_reduc is None:
        return X, y, X_train, X_test, y_train, y_test

    if dim_reduc == 'LDA':
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = reduce_lda(n, X_train, X_test, y_train, y_test)

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


def plot_2D_decision_boundaries(X, y, X_test, title, clf, folder=None, i=0, df_id=None, sub_dir_name=None):
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


def compute_model(X, y, n, clf=None, dim_reduc_name="LDA", resolution="10min", df_id=None, days=None):
    X_lda, y_lda, X_train, X_test, y_train, y_test = process_fold(2, X, y, n, dim_reduc=dim_reduc_name)
    print("fitting...")
    clf.fit(X_train, y_train)
    # clf = clf.best_estimator_
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    p_y_true, p_y_false = get_proba(y_probas, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true,\
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
            p_y_false*100, p_y_true*100,
            np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
            np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
            np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
            resolution))

    title = '%s-%s %dD %dFCV\nfold_i=%d, acc=%.1f%%, p0=%d%%, p1=%d%%, r0=%d%%, r1=%d%%, p0=%d%%, p1=%d%%\ndataset: class0=%d;' \
            'class1=%d\ntraining: class0=%d; class1=%d\ntesting: class0=%d; class1=%d\nresolution=%s\n' % (
                'LREG', '' if dim_reduc_name is None else dim_reduc_name, 2, 3, 0,
                acc * 100, precision_false * 100, precision_true * 100, recall_false * 100, recall_true * 100,
                p_y_false*100, p_y_true*100,
                np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1),
                np.count_nonzero(y_train == 0), np.count_nonzero(y_train == 1),
                np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                resolution)

    sub_dir_name = "days_%d_class0_%d_class1_%d" % (days, np.count_nonzero(y_lda == 0), np.count_nonzero(y_lda == 1))
    file_path = plot_2D_decision_boundaries(X_lda, y_lda, X_test, title, clf, folder='%s\\transition\\classifier_transit' % farm_id, i=n, df_id=df_id, sub_dir_name=sub_dir_name)
    print(n, acc)
    return acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name


def process_(df, days):
    print("process_...")
    print(df)
    # df1 = pd.DataFrame(cwt1)
    # df2 = pd.DataFrame(cwt2)
    # df1['label'] = 0
    # df2['label'] = 1
    # df = pd.concat([df1, df2], axis=0)
    labels = df["label"]
    df = df.drop('label', 1)
    n_slice = int(days/7)
    data_col_n = df.iloc[[0]].size
    chunk_size = int(data_col_n/n_slice)
    df.columns = df.columns.map(str)
    print(df)
    dfs = chunck_df(df, chunk_size, labels)
    data_acc, data_pf, data_pt, data_rf, data_rt, data_ff, data_ft, data_sf, data_st = {}, {}, {}, {}, {}, {}, {}, {}, {}
    cpt = 0
    sub_dir_name = None
    for id, data_frame in enumerate(dfs):
        # kf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
        # param_grid = {'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # clf = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), param_grid)
        clf = LogisticRegression()
        X, y = process_data_frame_(data_frame)
        acc_list, p_f_list, p_t_list, recall_f_list, recall_t_list, fscore_f_list, fscore_t_list, support_f_list, support_t_list = [], [], [], [], [], [], [], [], []
        N_ITER = 1000
        for n in range(N_ITER):
            try:
                acc, precision_false, precision_true, recall_false, recall_true, fscore_false, fscore_true, support_false, support_true, sub_dir_name = compute_model(X, y, n, clf=clf, df_id=id, days=days)
            except ValueError as e:
                print(e)
                continue

            acc_list.append(acc)
            p_f_list.append(precision_false)
            p_t_list.append(precision_true)
            recall_f_list.append(recall_false)
            recall_t_list.append(recall_true)
            fscore_f_list.append(fscore_false)
            fscore_t_list.append(fscore_true)
            support_f_list.append(support_false)
            support_t_list.append(support_true)
        print("acc_list", acc_list)
        data_acc[cpt] = acc_list
        data_pf[cpt] = p_f_list
        data_pt[cpt] = p_t_list
        data_rf[cpt] = recall_f_list
        data_rt[cpt] = recall_t_list
        data_ff[cpt] = fscore_f_list
        data_ft[cpt] = fscore_t_list
        data_sf[cpt] = support_f_list
        data_st[cpt] = support_t_list
        cpt += 1

    ribbon_plot_dir = '%s\\transition\\ribbon_transit\\%s' % (farm_id, sub_dir_name)
    pathlib.Path(ribbon_plot_dir).mkdir(parents=True, exist_ok=True)

    plot_(ribbon_plot_dir, data_acc, 'Classifier accuracy over time during increase of the FAMACHA score', "model accuracy in %")
    plot_(ribbon_plot_dir, data_pf, 'Classifier precision(False) over time during increase of the FAMACHA score', "model precision(False) in %")
    plot_(ribbon_plot_dir, data_pt, 'Classifier precision(True) over time during increase of the FAMACHA score', "model precision(True) in %")
    plot_(ribbon_plot_dir, data_rf, 'Classifier recall(False) over time during increase of the FAMACHA score', "model recall(False) in %")
    plot_(ribbon_plot_dir, data_rt, 'Classifier recall(True) over time during increase of the FAMACHA score', "model recall(True) in %")
    plot_(ribbon_plot_dir, data_ff, 'Classifier Fscore(False) over time during increase of the FAMACHA score', "model Fscore(False) in %")
    plot_(ribbon_plot_dir, data_ft, 'Classifier Fscore(True) over time during increase of the FAMACHA score', "model Fscore(True) in %")
    plot_(ribbon_plot_dir, data_sf, 'Classifier Support(False) over time during increase of the FAMACHA score', "model Support(False)")
    plot_(ribbon_plot_dir, data_st, 'Classifier Support(True) over time during increase of the FAMACHA score', "model Support(True)")
    
    fig = plt.figure(figsize=(10, 25))
    ax = fig.add_subplot(711)
    plot(ax, data_acc, 'Classifier accuracy over time during increase of the FAMACHA score', "model accuracy in %")
    ax = fig.add_subplot(712)
    plot(ax, data_pf, 'Classifier precision(False) over time during increase of the FAMACHA score', "model precision(False) in %")
    ax = fig.add_subplot(713)
    plot(ax, data_pt, 'Classifier precision(True) over time during increase of the FAMACHA score', "model precision(True) in %")
    ax = fig.add_subplot(714)
    plot(ax, data_rf, 'Classifier recall(False) over time during increase of the FAMACHA score', "model recall(False) in %")
    ax = fig.add_subplot(715)
    plot(ax, data_rt, 'Classifier recall(True) over time during increase of the FAMACHA score', "model recall(True) in %")
    ax = fig.add_subplot(716)
    plot(ax, data_ff, 'Classifier Fscore(False) over time during increase of the FAMACHA score', "model Fscore(False) in %")
    ax = fig.add_subplot(717)
    plot(ax, data_ft, 'Classifier Fscore(True) over time during increase of the FAMACHA score', "model Fscore(True) in %")
    # ax = fig.add_subplot(918)
    # plot(ax, data_sf, 'Classifier Support(False) over time during increase of the FAMACHA score', "model Support(False)")
    # ax = fig.add_subplot(919)
    # plot(ax, data_st, 'Classifier Support(True) over time during increase of the FAMACHA score', "model Support(True)")
    plt.tight_layout()
    plt.show()
    path = "%s\\model_transit.png" % ribbon_plot_dir
    print(path)
    fig.savefig(path, dpi=100)
    fig.clear()
    plt.close(fig)


def plot_(path, data, title, y_label):
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
    ax.set_title(title)
    # ax = df.copy().plot.box(grid=True, patch_artist=True, title=title, figsize=(10, 7))
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
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


def process_data_frame(data_frame, data_frame_0, out_fname=None, df_hum=None, df_temp=None):
    global DATA_
    DATA_ = []
    print(out_fname)
    # data_frame = data_frame.fillna(-1)
    X = data_frame[data_frame.columns[0:data_frame.shape[1] - 1]].values
    X_t = df_temp[df_temp.columns[0:df_temp.shape[1] - 1]].values
    X_h = df_hum[df_hum.columns[0:df_hum.shape[1] - 1]].values
    X_date = data_frame_0[data_frame_0.columns[data_frame_0.shape[1] - 7:data_frame_0.shape[1]]].values
    # cwt_list = []
    class0 = []
    class1 = []
    H = []
    data_frame_0 = data_frame_0.reset_index(drop=True)

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
    print("finished calculating herd cwt.")
    # herd_mean = minmax_scale(herd_mean, feature_range=(0, 1))

    purge_file(out_fname)
    X_cwt = pd.DataFrame()
    cpt = 0
    with open(out_fname, 'a') as outfile:
        for activity, (i, row) in zip(X, data_frame_0.iterrows()):
            meta = row["label":].values.tolist()
            print(meta)
            # activity = item
            # temperature = item[1].tolist()
            # humidity = item[2].tolist()
            print("interpolate...")
            activity = interpolate(activity)
            activity = np.asarray(activity)
            print("divide...")
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
            print(coefs.shape)

            if len(DATA_) < 1:
                DATA_.append({'coef_shape': coefs_herd_mean.shape, 'freqs': freqs_h})
                print(DATA_)

            if cpt == 0:
                X_cwt = pd.DataFrame(columns=[str(x) for x in range(len(cwt))], dtype=np.float16)

            X_cwt.loc[cpt] = cwt
            print("appending df...")
            cpt += 1
            # print(X_cwt)
            target = data_frame.at[i, 'label']

            label = 'False'
            if target == 0:
                class0.append(activity)
            if target == 1:
                label = 'True'
                class1.append(activity)

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
    return X.values, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean, coefs_herd_mean, herd_mean


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


def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y == c, :], axis=0) * feature_importances)
        )

    return out


def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr


def normalized(v):
    return v / np.sqrt(np.sum(v ** 2))


def process(data_frame, data_frame_0, out_fname=None, df_hum=None, df_temp=None, resolution=None,
            days=None, f_config=None, farm_id=None, out_dir=None):
    global DATA_
    plt.clf()
    print("process...", resolution, days)
    X, y, scales, delta_t, wavelet_type, class0_mean, coefs_class0_mean, class1_mean, coefs_class1_mean,\
    coefs_herd_mean, herd_mean = process_data_frame(
        data_frame, data_frame_0, out_fname, df_hum=df_hum, df_temp=df_temp)
    print("train_test_split...")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    clf = SVC(kernel='linear')
    # clf = LDA(n_components=2)
    y_train[0] = 3

    print("fit...")

    clf.fit(X_train, y_train)
    del X_train
    del y_train
    gc.collect()

    # result = class_feature_importance(X.values, y, clf.coef_[0])
    print("explain_prediction...")
    aux1 = eli5.sklearn.explain_prediction.explain_prediction_linear_classifier(clf, X[0], top=X.shape[1])
    aux1 = eli5.format_as_dataframe(aux1)
    print("********************************")
    print(aux1)
    class0 = aux1[aux1.target == 0]
    class1 = aux1[aux1.target == 1]
    del aux1

    class0 = class0[class0.feature != '<BIAS>']
    class1 = class1[class1.feature != '<BIAS>']

    class0['feature'] = class0['feature'].str.replace('x', '')
    class1['feature'] = class1['feature'].str.replace('x', '')

    class0['feature'] = class0['feature'].apply(int)
    class1['feature'] = class1['feature'].apply(int)

    class0 = class0.sort_values('feature')
    class1 = class1.sort_values('feature')

    weight0 = class0['weight'].values
    weight1 = class1['weight'].values

    del class1
    weight0 = pad(weight0, DATA_[0]['coef_shape'][0] * DATA_[0]['coef_shape'][1])
    weight1 = pad(weight1, DATA_[0]['coef_shape'][0] * DATA_[0]['coef_shape'][1])

    filename = "weight_data.json"
    try:
        os.remove(filename)
    except OSError:
        pass
    print("saving weights...")
    with open(filename, 'w') as f:
        w_data = {"weight0": weight0.tolist(), "weight1": weight1.tolist(),
                  "shape": DATA_[0]['coef_shape']}
        json.dump(w_data, f)

    print("building figure...")
    if len(class0) > 0:
        del class0

        fig, axs = plt.subplots(5, 2)
        fig.set_size_inches(15, 15)

        axs[0, 0].set_title(out_fname, fontsize=25, loc="left", pad=30)
        ymin = min([min(class0_mean), min(class1_mean), min(herd_mean)])
        ymax = max([max(class0_mean), max(class1_mean), max(herd_mean)])

        x_axis = [x for x in range(coefs_class0_mean.shape[1])]
        # x_axis[0] = 1
        # x_axis[-1] = coefs_class0_mean.shape[1]
        axs[0, 0].pcolormesh(x_axis, DATA_[0]['freqs'], coefs_class0_mean)

        axs[0, 0].set_yscale('log')
        axs[0, 0].set_title('class0 cwt input')
        # iwave = wavelet.icwt(coefs_class0_mean, scales, delta_t, wavelet=wavelet_type)
        # iwave = np.real(iwave)
        axs[0, 1].plot(class0_mean)
        del class0_mean
        axs[0, 1].set_ylim([ymin, ymax])
        axs[0, 1].set_title('class0 time input')

        axs[1, 0].pcolormesh(x_axis, DATA_[0]['freqs'], coefs_class1_mean)

        axs[1, 0].set_yscale('log')
        axs[1, 0].set_title('class1 cwt input')
        # iwave = wavelet.icwt(coefs_class1_mean, scales, delta_t, wavelet=wavelet_type)
        # iwave = np.real(iwave)
        axs[1, 1].plot(class1_mean)
        del class1_mean
        axs[1, 1].set_ylim([ymin, ymax])
        axs[1, 1].set_title('class1 time input')

        c0 = np.reshape(normalized(weight0), DATA_[0]['coef_shape'])
        del weight0
        print("computing icwt of weight0")
        iwave0 = wavelet.icwt(c0, scales, delta_t, wavelet=wavelet_type)
        iwave0 = np.real(iwave0)

        c1 = np.reshape(normalized(weight1), DATA_[0]['coef_shape'])
        del weight1
        print("computing icwt of weight1")
        iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
        iwave1 = np.real(iwave1)

        ymin2 = min([min(iwave0), min(iwave1)])
        ymax2 = max([max(iwave0), max(iwave1)])
        print("pcolor...")
        axs[2, 0].pcolormesh(x_axis, DATA_[0]['freqs'], c0)
        axs[2, 0].set_yscale('log')
        axs[2, 0].set_title('class0 cwt weight')
        axs[2, 1].plot(iwave0)
        del iwave0
        axs[2, 1].set_ylim([ymin2, ymax2])
        axs[2, 1].set_title('class0 cwt weight inverse')

        axs[3, 0].pcolormesh(x_axis, DATA_[0]['freqs'], c1)
        axs[3, 0].set_yscale('log')
        axs[3, 0].set_title('class1 cwt weight')
        axs[3, 1].plot(iwave1)
        del iwave1
        axs[3, 1].set_ylim([ymin2, ymax2])
        axs[3, 1].set_title('class1 cwt weight inverse')

        axs[4, 0].pcolormesh(x_axis, DATA_[0]['freqs'], coefs_herd_mean)
        del coefs_herd_mean
        # Set yscale, ylim and labels
        axs[4, 0].set_yscale('log')
        axs[4, 0].set_title('mean cwt input')
        # iwave_m = wavelet.icwt(coefs_herd_mean, scales, delta_t, wavelet=wavelet_type)
        # iwave_m = np.real(iwave_m)
        axs[4, 1].set_ylim([ymin, ymax])
        axs[4, 1].plot(herd_mean)
        del herd_mean
        axs[4, 1].set_title('mean time input')
        gc.collect()
        del coefs_class1_mean
        del coefs_class0_mean
        # return coefs_class0_mean, coefs_class1_mean
    else:
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(50, 25)
        ymin = min([min(class1_mean), min(herd_mean)])
        ymax = max([max(class1_mean), max(herd_mean)])

        axs[0, 0].set_title(
            out_fname + ' ' + resolution + ' ' + str(days) + ' ' + str(f_config) + ' ' + str(data_frame.shape),
            fontsize=25, loc="left", pad=30)

        x_axis = [x for x in range(coefs_class1_mean.shape[1])]
        # axs[0, 0].pcolor(x_axis, DATA_[0]['freqs'], coefs_class1_mean)
        # axs[0, 0].set_yscale('log')
        # axs[0, 0].set_title('class1 cwt input')
        # axs[0, 1].plot(class1_mean)
        # axs[0, 1].set_ylim([ymin, ymax])
        # axs[0, 1].set_title('class1 time input')

        c1 = np.reshape(normalized(weight1), DATA_[0]['coef_shape'])
        print("computin icwt of weight1")
        iwave1 = wavelet.icwt(c1, scales, delta_t, wavelet=wavelet_type)
        iwave1 = np.real(iwave1)

        ymin2 = min(iwave1)
        ymax2 = max(iwave1)

        axs[0, 0].pcolormesh(x_axis, DATA_[0]['freqs'], c1)
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_title('class1 cwt weight')
        axs[0, 1].plot(iwave1)
        del iwave1
        axs[0, 1].set_ylim([ymin2, ymax2])
        axs[0, 1].set_title('class1 cwt weight inverse')

        axs[1, 0].pcolormesh(x_axis, DATA_[0]['freqs'], coefs_herd_mean)
        del coefs_herd_mean
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_title('mean cwt input')

        axs[1, 1].set_ylim([ymin, ymax])
        axs[1, 1].plot(herd_mean)
        del herd_mean
        axs[1, 1].set_title('mean time input')

    fig.show()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    outfile = '%s\\%s_SVC_%s_%s_days_%d_%s_%d_%d.png' % (out_dir, farm_id, out_fname.split('.')[0], resolution, days,
                                                str(f_config).replace(',', '').replace('[', '').replace(']', ''),
                                                data_frame.shape[0], data_frame.shape[1])
    fig.savefig(outfile, dpi=100)
    fig.clear()
    plt.close(fig)
    DATA_ = []


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


if __name__ == '__main__':
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
        for farm_id in ["cedara_70091100056"]:
            # try:
            #     shutil.rmtree("transition")
            # except (OSError, FileNotFoundError) as e:
            #     print(e)

            for days in [4]:
                dir = TRAINING_DIR + '%s_sld_0_dbt%d_%s/' % (resolution, days, farm_id)
                # os.chdir(dir)

                # start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_cwt_div.data' % (days, farm_id),
                #       resolution=resolution,
                #       days=days,
                #       farm_id=farm_id,
                #       filter=False,
                #       f_config=[1, 2],
                #       fname_temp="%s/training_sets/temperature.data" % dir,
                #       fname_hum="%s/training_sets/humidity.data" % dir)
                #
                # exit()
                start(fname="%s/training_sets/activity_.data" % dir, out_fname='%d_%s_cwt_div.data' % (days, farm_id),
                      resolution=resolution,
                      days=days,
                      farm_id=farm_id,
                      f_config=[1, 2],
                      filter_delmas=False,
                      filter_cedara=True,
                      fname_temp="%s/training_sets/temperature.data" % dir,
                      fname_hum="%s/training_sets/humidity.data" % dir,
                      output_clf_transit=False
                      )
                start(fname="%s/training_sets/cwt_.data" % dir, out_fname=None,
                      resolution=resolution,
                      days=days,
                      farm_id=farm_id,
                      f_config=[1, 2],
                      filter_delmas=False,
                      filter_cedara=True,
                      fname_temp="%s/training_sets/temperature.data" % dir,
                      fname_hum="%s/training_sets/humidity.data" % dir,
                      output_clf_transit=True
                      )
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
