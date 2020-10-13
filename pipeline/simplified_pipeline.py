from __future__ import division  # for python2 regular div

import shutil

import matplotlib
import glob2
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from datetime import datetime
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

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


def load_df_from_datasets(fname, label_col='label'):
    print("load_df_from_datasets...", fname)
    # df = pd.read_csv(fname, nrows=1, sep=",", header=None, error_bad_lines=False)
    # # print(df)
    # type_dict = find_type_for_mem_opt(df)

    data_frame = pd.read_csv(fname, sep=",", header=None, low_memory=False)
    print("shape before removal of duplicates=", data_frame.shape)
    data_frame = data_frame.drop_duplicates()
    print("shape after removal of duplicates=", data_frame.shape)
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
    data_frame = shuffle(data_frame)
    return data_frame_original, data_frame, cols_to_keep


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
    pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
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
        pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))

        print("->MinMaxScaler->SVC")
        pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True, class_weight='balanced'))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        print(str(classification_report(y_test, y_pred, output_dict=True)))

        print("->StandardScaler->LDA(1)->SVC")
        pipe = Pipeline(
            [('scaler', preprocessing.StandardScaler()), ('lda', LDA(n_components=1)), ('svc', SVC(probability=True, class_weight='balanced'))])
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
    data_iris = data_iris[data_iris.target != 2.0] # remove class 2
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
    print(lo_x_boot, hi_x_boot)
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
        dir_path += sub_dir+"/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


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


def plot_2d_space(X, y, filename_2d_scatter, label='Classes'):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        ax.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    ax.title(label)
    ax.legend(loc='upper right')
    print(filename_2d_scatter)
    fig.savefig(filename_2d_scatter)

    plt.show()
    plt.close(fig)
    plt.clf()


def process_data_frame(out_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats, sampling, downsample_false_class, y_col='label'):
    print("*******************************************************************")
    print("downsample_false_class=", downsample_false_class)
    print("*******************************************************************")
    report_rows_list = []
    if downsample_false_class:
        df_true = data_frame[data_frame['label'] == True]
        df_false = data_frame[data_frame['label'] == False]
        df_false = df_false.sample(df_true.shape[0])
        data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)

    data_frame = data_frame.dropna()
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - 1]]

    filename_2d_scatter = "%s/%s_2DPCA_days_%d_threshi_%d_threshz_%d_option_%s_downsampled_%s_sampling_%s.png" % (
        output_dir, farm_id, days, thresh_i, thresh_z, option, downsample_false_class, sampling)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    if not os.path.exists(output_dir):
        print("mkdir", output_dir)
        os.makedirs(output_dir)
    plot_2d_space(X, y, filename_2d_scatter, '(2 PCA components)')

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
        'recall_score1': make_scorer(recall_score,  average=None, labels=[1]),
        'f1_score0': make_scorer(f1_score, average=None, labels=[0]),
        'f1_score1': make_scorer(f1_score, average=None, labels=[1])
    }

    param_str = "option_%s_downsample_%s_threshi_%d_threshz_%d_days_%d_farmid_%s_nrepeat_%d_nsplits_%d_class0_%s_class1_%s_sampling_%s" % (option, str(downsample_false_class), thresh_i, thresh_z, days, farm_id, n_repeats, n_splits, class0_count, class1_count, sampling)

    print('->SVC')
    clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    cv_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
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
    make_roc_curve(out_dir, clf_svc, X, y, cv_svc, param_str, thresh_i, thresh_z)
    del scores
    
    print('->LDA')
    clf_lda1_svc = make_pipeline(LDA())
    cv_lda1_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
    scores = cross_validate(clf_lda1_svc, X.copy(), y.copy(), cv=cv_lda1_svc, scoring=scoring, n_jobs=-1)
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
    scores["classifier"] = "->LDA"
    scores["classifier_details"] = str(clf_lda1_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    make_roc_curve(out_dir, clf_lda1_svc, X, y, cv_lda1_svc, param_str, thresh_i, thresh_z)
    del scores

    print('->LDA(1)->SVC')
    clf_lda = make_pipeline(LDA(n_components=1), SVC(probability=True, class_weight='balanced'))
    cv_lda = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
    scores = cross_validate(clf_lda, X.copy(), y.copy(), cv=cv_lda, scoring=scoring, n_jobs=-1)
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
    scores["classifier"] = "->LDA"
    scores["classifier_details"] = str(clf_lda).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    make_roc_curve(out_dir, clf_lda, X, y, cv_lda, param_str, thresh_i, thresh_z)
    del scores

    print('->StandardScaler->SVC')
    clf_std_svc = make_pipeline(preprocessing.StandardScaler(), SVC(probability=True, class_weight='balanced'))
    cv_std_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
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
    make_roc_curve(out_dir, clf_std_svc, X, y, cv_std_svc, param_str, thresh_i, thresh_z)
    del scores

    print('->MinMaxScaler->SVC')
    clf_minmax_svc = make_pipeline(preprocessing.MinMaxScaler(), SVC(probability=True, class_weight='balanced'))
    cv_minmax_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
    scores = cross_validate(clf_minmax_svc, X.copy(), y.copy(), cv=cv_minmax_svc, scoring=scoring, n_jobs=-1)
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
    scores["classifier"] = "->MinMaxScaler->SVC"
    scores["classifier_details"] = str(clf_minmax_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    make_roc_curve(out_dir, clf_minmax_svc, X, y, cv_minmax_svc, param_str, thresh_i, thresh_z)
    del scores
    
    print('->Normalize(l2)->SVC')
    clf_normalize_svc = make_pipeline(preprocessing.Normalizer(), SVC(probability=True, class_weight='balanced'))
    cv_normalize_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
    scores = cross_validate(clf_normalize_svc, X.copy(), y.copy(), cv=cv_normalize_svc, scoring=scoring, n_jobs=-1)
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
    scores["classifier"] = "->Normalize(l2)->SVC"
    scores["classifier_details"] = str(clf_normalize_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    make_roc_curve(out_dir, clf_normalize_svc, X, y, cv_normalize_svc, param_str, thresh_i, thresh_z)
    del scores
    
    print('->Normalize(l2)->MinMaxScaler->SVC')
    clf_normalize_minmax_svc = make_pipeline(preprocessing.Normalizer(), preprocessing.MinMaxScaler(), SVC(probability=True, class_weight='balanced'))
    cv_normalize_minmax_svc = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=int(datetime.now().microsecond / 10))
    scores = cross_validate(clf_normalize_minmax_svc, X.copy(), y.copy(), cv=cv_normalize_minmax_svc, scoring=scoring, n_jobs=-1)
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
    scores["classifier"] = "->Normalize(l2)->SVC"
    scores["classifier_details"] = str(clf_normalize_minmax_svc).replace('\n', '').replace(" ", '')
    report_rows_list.append(scores)
    make_roc_curve(out_dir, clf_normalize_minmax_svc, X, y, cv_normalize_minmax_svc, param_str, thresh_i, thresh_z)
    del scores


    # print('->Normalize(l2)->RandomForestClassifier')
    # clf_normalize_random = make_pipeline(preprocessing.Normalizer(), RandomForestClassifier())
    # cv_normalize_random = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                              random_state=int(datetime.now().microsecond / 10))
    # scores = cross_validate(clf_normalize_random, X.copy(), y.copy(), cv=cv_normalize_random, scoring=scoring, n_jobs=-1)
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
    # scores["classifier"] = "->Normalize(l2)->RandomForestClassifier"
    # scores["classifier_details"] = str(clf_normalize_random).replace('\n', '').replace(" ", '')
    # report_rows_list.append(scores)
    # make_roc_curve(out_dir, clf_normalize_random, X, y, cv_normalize_random, param_str, thresh_i, thresh_z)
    # del scores


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
        n_process = int(sys.argv[5])

    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_folder)
    print("n_splits=", n_splits)
    print("n_repeats=", n_repeats)
    print("n_process=", n_process)
    print("loading dataset...")

    # if os.path.exists(output_dir):
    #     print("purge %s..." % output_dir)
    #     try:
    #         shutil.rmtree(output_dir)
    #     except IOError:
    #         print("file not found.")

    files = glob2.glob(dataset_folder)
    files = [file.replace("\\", '/') for file in files]
    print("found %d files." % len(files))

    MULTI_THREADING_ENABLED = (n_process > 0)
    print("MULTI_THREADING_ENABLED=", MULTI_THREADING_ENABLED)

    if MULTI_THREADING_ENABLED:
        pool = Pool(processes=n_process)
        for file in files:
            data_frame_original, data_frame, _ = load_df_from_datasets(file)
            thresh_i, thresh_z, days, farm_id, option, sampling = parse_param_from_filename(file)
            print("thresh_i=", thresh_i)
            print("thresh_z=", thresh_z)
            print("days=", days)
            print("farm_id=", farm_id)
            print("option=", option)
            pool.apply_async(process_data_frame,
                             (output_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats, sampling, True,))
            pool.apply_async(process_data_frame,
                             (output_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats, sampling, False,))
        pool.close()
        pool.join()
        pool.terminate()
    else:
        for file in files:
            data_frame_original, data_frame, _ = load_df_from_datasets(file)
            thresh_i, thresh_z, days, farm_id, option, sampling = parse_param_from_filename(file)
            print("thresh_i=", thresh_i)
            print("thresh_z=", thresh_z)
            print("days=", days)
            print("farm_id=", farm_id)
            print("option=", option)
            process_data_frame(output_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats,
                               sampling, False)
            process_data_frame(output_dir, data_frame, thresh_i, thresh_z, days, farm_id, option, n_splits, n_repeats,
                               sampling, True)

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
