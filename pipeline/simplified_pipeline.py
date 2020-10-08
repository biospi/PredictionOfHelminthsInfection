from __future__ import division #for python2 regular div
import matplotlib
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
    df = pd.read_csv(fname, nrows=1, sep=",", header=None, error_bad_lines=False)
    # print(df)
    type_dict = find_type_for_mem_opt(df)

    data_frame = pd.read_csv(fname, sep=",", header=None, dtype=type_dict, low_memory=False, error_bad_lines=False)
    data_frame = data_frame.drop_duplicates()
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

    data_frame.columns = hearder
    data_frame_original = data_frame.copy()
    cols_to_keep = hearder[:-META_DATA_LENGTH]
    cols_to_keep.append(label_col)
    data_frame = data_frame[cols_to_keep]
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
    pipe = Pipeline([('svc', SVC(probability=True))])
    pipe.fit(X1.copy(), y1.copy())
    y_pred = pipe.predict(X2.copy())
    print(classification_report(y2, y_pred))

    print("->StandardScaler->SVC")
    pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True))])
    pipe.fit(X1.copy(), y1.copy())
    y_pred = pipe.predict(X2.copy())
    print(classification_report(y2, y_pred))

    print("->MinMaxScaler->SVC")
    pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True))])
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
        outfile.write('->SVC\n')
        pipe = Pipeline([('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->LDA")
        outfile.write('->LDA\n')
        pipe = Pipeline([('lda', LDA())])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->StandardScaler->SVC")
        outfile.write("->StandardScaler->SVC\n")
        pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->MinMaxScaler->SVC")
        outfile.write("->MinMaxScaler->SVC\n")
        pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->StandardScaler->LDA(1)->SVC")
        outfile.write("->StandardScaler->LDA(1)->SVC\n")
        pipe = Pipeline(
            [('scaler', preprocessing.StandardScaler()), ('lda', LDA(n_components=1)), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->LDA(1)->SVC")
        outfile.write("->LDA(1)->SVC\n")
        pipe = Pipeline([('reduce_dim', LDA(n_components=1)), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')


def process_data_frame(data_frame, output_dir, test_size, thresh_i, thresh_z, days, farm_id, option, y_col='label', downsample_false_class=True):
    print("*******************************************************************")
    print("downsample_false_class=", downsample_false_class)
    print("*******************************************************************")
    if downsample_false_class:
        df_true = data_frame[data_frame['label'] == True]
        df_false = data_frame[data_frame['label'] == False]
        df_false = df_false.head(df_true.shape[0])

        data_frame = pd.concat([df_true, df_false], ignore_index=True, sort=False)
    data_frame = data_frame.dropna()
    y = data_frame[y_col].values.flatten()
    y = y.astype(int)
    X = data_frame[data_frame.columns[2:data_frame.shape[1] - 1]]
    print("test_size=", test_size, test_size/100)
    t_s = float(test_size/100)
    print("test size in percent=", t_s)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=t_s)
    print("training", "class0=", y_train[y_train == 0].size, "class1=", y_train[y_train == 1].size)
    print("test", "class0=", y_test[y_test == 0].size, "class1=", y_test[y_test == 1].size)

    # plt.hist(X_train.values.flatten(), bins='auto', histtype='step', density=True)
    # plt.title("Distribution of training data")
    # plt.show()

    try:
        pathlib.Path(output_dir).mkdir(parents=True)
    except Exception as e:
        print(e)
        
    filename = "%s/%s_classification_report_days_%d_threshi_%d_threshz_%d_testsize_%d_%s.txt" % (output_dir, farm_id, days, thresh_i, thresh_z, test_size, option)
    with open(filename, 'a') as outfile:
        outfile.write("************************************************\n")
        outfile.write("downsample on= " + str(downsample_false_class)+"\n")
        outfile.write("training-> class0="+ str(y_train[y_train == 0].size) + " class1=" + str(y_train[y_train == 1].size)+"\n")
        outfile.write("test-> class0="+ str(y_test[y_test == 0].size) + " class1=" + str(y_test[y_test == 1].size)+"\n")

        print("->SVC")
        outfile.write('->SVC\n')
        pipe = Pipeline([('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        # print("->LDA")
        # outfile.write('->LDA')
        # pipe = Pipeline([('lda', LDA())])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, y_pred))

        # print("->PLSRegression(10)")
        # outfile.write('->SVC')
        # pipe = Pipeline([('pls', PLSRegression(n_components=10))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, np.round(y_pred)))
        #
        # print("->PLSRegression(100)")
        # pipe = Pipeline([('pls', PLSRegression(n_components=100))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, np.round(y_pred)))
        #
        # print("->StandardScaler->PLSRegression(10)")
        # pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('pls', PLSRegression(n_components=10))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, np.round(y_pred)))

        print("->StandardScaler->SVC")
        outfile.write("->StandardScaler->SVC\n")
        pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')


        print("->MinMaxScaler->SVC")
        outfile.write("->MinMaxScaler->SVC\n")
        pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, np.round(y_pred)))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->StandardScaler->LDA(1)->SVC")
        outfile.write("->StandardScaler->LDA(1)->SVC\n")
        pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('lda', LDA(n_components=1)), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        print("->LDA(1)->SVC")
        outfile.write("->LDA(1)->SVC\n")
        pipe = Pipeline([('reduce_dim', LDA(n_components=1)), ('svc', SVC(probability=True))])
        pipe.fit(X_train.copy(), y_train.copy())
        y_pred = pipe.predict(X_test.copy())
        print(classification_report(y_test, y_pred))
        outfile.write(str(classification_report(y_test, y_pred, output_dict=True)))
        outfile.write('\n\n')

        # print("->LDA(1)->LDA")
        # outfile.write("->LDA(1)->LDA")
        # pipe = Pipeline([('lda', LDA(n_components=1)), ('lda_clf', LDA())])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, y_pred))

        # print("->PCA(1)->SVC")
        # pipe = Pipeline([('pca', PCA(n_components=1)), ('svc', SVC(probability=True))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, y_pred))
        #
        # print("->PCA(10)->SVC")
        # pipe = Pipeline([('pca', PCA(n_components=10)), ('svc', SVC(probability=True))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, y_pred))
        #
        # print("->PCA(30)->SVC")
        # pipe = Pipeline([('pca', PCA(n_components=30)), ('svc', SVC(probability=True))])
        # pipe.fit(X_train.copy(), y_train.copy())
        # y_pred = pipe.predict(X_test.copy())
        # print(classification_report(y_test, y_pred))

    # print("*******************************************")
    # print("STEP BY STEP")
    # print("*******************************************")
    #
    # clf_lda = LDA(n_components=1)
    # X_train_r = clf_lda.fit_transform(X_train.copy(), y_train.copy())
    # X_test_r = clf_lda.transform(X_test.copy())
    #
    # X_reduced = np.concatenate((X_train_r.copy(), X_test_r.copy()), axis=0)
    # y_reduced = np.concatenate((y_train.copy(), y_test.copy()), axis=0)
    #
    # print("->LDA(1)->SVC")
    # plot_2D_decision_boundaries(SVC(probability=True), "svc", "dim_reduc_name", 1, 1, "",
    #                             X_reduced.copy(),
    #                             y_reduced.copy(),
    #                             X_test_r.copy(),
    #                             y_test.copy(),
    #                             X_train_r.copy(),
    #                             y_train.copy())
    # print("->LDA(1)->LDA")
    # plot_2D_decision_boundaries(LDA(), "lda", "dim_reduc_name", 1, 1, "",
    #                             X_reduced.copy(),
    #                             y_reduced.copy(),
    #                             X_test_r.copy(),
    #                             y_test.copy(),
    #                             X_train_r.copy(),
    #                             y_train.copy())

    # clf_pls = PLSRegression(n_components=1)
    # X_train_r = clf_pls.fit_transform(X_train.copy(), y_train.copy())[0]
    # X_test_r = clf_pls.transform(X_test.copy())
    #
    # X_reduced = np.concatenate((X_train_r, X_test_r), axis=0)
    # y_reduced = np.concatenate((y_train, y_test), axis=0)
    #
    # print("->PLS(1)->SVC")
    # plot_2D_decision_boundaries(SVC(probability=True), "svc", "dim_reduc_name", 1, 1, "",
    #                             X_reduced.copy(),
    #                             y_reduced.copy(),
    #                             X_test_r.copy(),
    #                             y_test.copy(),
    #                             X_train_r.copy(),
    #                             y_train.copy())
    # print("->PLS(1)->LDA")
    # plot_2D_decision_boundaries(LDA(), "lda", "dim_reduc_name", 1, 1, "",
    #                             X_reduced.copy(),
    #                             y_reduced.copy(),
    #                             X_test_r.copy(),
    #                             y_test.copy(),
    #                             X_train_r.copy(),
    #                             y_train.copy())


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


if __name__ == "__main__":
    print("args: output_dir dataset_filepath test_size")
    # print("********************************************************************")
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]
    # y = iris.target
    # dummy_run(X, y, 40, "dummy_iris.txt")
    # print("********************************************************************")
    # X, y = make_blobs(n_samples=50, centers=2, n_features=100, center_box=(0, 10))
    # dummy_run(X, y, 40, "dummy_blob.txt")
    # print("********************************************************************")
    # exit(0)

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        dataset_filepath = sys.argv[2]
        test_size = int(sys.argv[3])
    else:
        exit(-1)

    print("output_dir=", output_dir)
    print("dataset_filepath=", dataset_filepath)
    print("test_size=", test_size)

    print("loading dataset...")
    data_frame_original, data_frame, _ = load_df_from_datasets(dataset_filepath)
    split = dataset_filepath.split("/")[-1].split('.')[0].split('_')
    thresh_i = int(split[-3])
    thresh_z = int(split[-1])
    days = int(split[4])
    farm_id = split[1] + "_" + split[2]
    option = split[0]
    print("thresh_i=", thresh_i)
    print("thresh_z=", thresh_z)
    print("days=", days)
    print("farm_id=", farm_id)
    print("option=", option)
    data_frame = shuffle(data_frame)
    process_data_frame(data_frame, output_dir, test_size, thresh_i, thresh_z, days, farm_id, option, downsample_false_class=True)
    process_data_frame(data_frame, output_dir, test_size, thresh_i, thresh_z, days, farm_id, option,  downsample_false_class=False)

    # exit(-1)
    # dataset_filepath1 = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\csv_db\\cedara_70091100056_720\\7_1\\training_sets\\cwt_.data"
    # dataset_filepath2 = "E:\\Users\\fo18103\\PycharmProjects\\prediction_of_helminths_infection\\training_data_generator_and_ml_classifier\\src\\csv_db\\delmas_70101200027_720\\7_1\\training_sets\\cwt_.data"
    #
    # _, data_frame1, _ = load_df_from_datasets(dataset_filepath1)
    # _, data_frame2, _ = load_df_from_datasets(dataset_filepath2)
    # process_cross_farm(data_frame1, data_frame2)


