import warnings
import keras
import pywt
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
import pycwt as wavelet
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras import metrics
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from tqdm import tqdm

from utils.Utils import create_rec_dir
from utils._custom_split import StratifiedLeaveTwoOut
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc
from sklearn.preprocessing import binarize, MinMaxScaler

from utils._cwt import cwt_power
from utils.visualisation import mean_confidence_interval, plot_roc_range
from keras.utils.vis_utils import plot_model
import time
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'


class Score:
    def __init__(self, y, class_healthy, class_unhealthy, steps, days, farm_id, test_balanced_accuracy_score,
                 test_precision_score0, test_precision_score1, test_recall_score0, test_recall_score1,
                 test_f1_score0, test_f1_score1, sampling, mean_auc, label_series,
                 cross_validation_method, start_time, output_dir, downsample_false_class, clf_detail=None):
        report_rows_list = []
        scores = {}
        scores["fit_time"] = 0
        scores["score_time"] = 0
        scores["test_balanced_accuracy_score"] = test_balanced_accuracy_score
        scores["test_precision_score0"] = test_precision_score0
        scores["test_precision_score1"] = test_precision_score1
        scores["test_recall_score0"] = test_recall_score0
        scores["test_recall_score1"] = test_recall_score1
        scores["test_f1_score0"] = test_f1_score0
        scores["test_f1_score1"] = test_f1_score1
        scores["downsample"] = downsample_false_class
        scores["class0"] = y[y == class_healthy].size
        scores["class1"] = y[y == class_unhealthy].size
        scores["steps"] = steps
        scores["days"] = days
        scores["farm_id"] = farm_id
        scores["balanced_accuracy_score_mean"] = np.mean(test_balanced_accuracy_score)
        scores["precision_score0_mean"] = np.mean(test_precision_score0)
        scores["precision_score1_mean"] = np.mean(test_precision_score1)
        scores["recall_score0_mean"] = np.mean(test_recall_score0)
        scores["recall_score1_mean"] = np.mean(test_recall_score1)
        scores["f1_score0_mean"] = np.mean(test_f1_score0)
        scores["f1_score1_mean"] = np.mean(test_f1_score1)
        scores["sampling"] = sampling
        scores["classifier"] = "->CNN"
        scores["classifier_details"] = clf_detail
        scores["roc_auc_score_mean"] = mean_auc
        report_rows_list.append(scores)

        df_report = pd.DataFrame(report_rows_list)
        df_report["class_0_label"] = label_series[class_healthy]
        df_report["class_1_label"] = label_series[class_unhealthy]
        df_report["nfold"] = cross_validation_method.nfold if hasattr(cross_validation_method, 'nfold') else np.nan
        df_report["total_fit_time"] = [time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))]
        filename = "%s/%s_classification_report_days_%d_option_%s_downsampled_%s_sampling_%s_%s.csv" % (
            output_dir, farm_id, days, steps, downsample_false_class, sampling, clf_detail)
        if not os.path.exists(output_dir):
            print("mkdir", output_dir)
            os.makedirs(output_dir)
        filename = filename.replace("->", "_")
        df_report.to_csv(filename, sep=',', index=False)
        print("filename=", filename)


def plot_roc(ax, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    y_pred_proba = y_pred[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=None, alpha=0.3, lw=1, c="tab:blue")
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred_idx, output_dict=True)
    print(report)
    targets = list(report.keys())[:-3]
    if len(targets) == 1:
        warnings.warn("classifier only predicted 1 target.")
    #     if targets[0] == "0":
    #         report["1"] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    #     if targets[0] == "1":
    #         report["0"] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    # print(report)
    # if np.isnan(roc_auc):
    #     roc_auc = 0
    print("roc_auc=", roc_auc)
    p0, p1, r0, r1, f0, f1 = 0, 0, 0, 0, 0, 0
    acc = report["accuracy"]
    if "0" in report:
        p0 = report["0"]["precision"]
        r0 = report["0"]["recall"]
        f0 = report["0"]["f1-score"]

    if "1" in report:
        p1 = report["1"]["precision"]
        r1 = report["1"]["recall"]
        f1 = report["1"]["f1-score"]

    return roc_auc, fpr, tpr, report, acc, p0, p1, r0, r1, f0, f1


def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def evaluate_model(i, out_dir, ax, trainX, trainy, testX, testy, verbose=0, epochs=1000, batch_size=8):
    X_train, X_test, y_train, y_test = format(trainX, trainy, testX, testy)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
    ]
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    # fit network
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['auc'])
    ##plt.plot(history.history['val_auc'])
    plt.title('model auc for fold %d' % i)
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    path = "%s/model/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'fold%d_modelauc.png' % (i))
    print(final_path)
    plt.savefig(final_path)
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    ##plt.plot(history.history['val_loss'])
    plt.title('model loss for fold %d' % i)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    path = "%s/model/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'fold%d_modelloss.png' % (i))
    print(final_path)
    plt.savefig(final_path)

    # evaluate model
    baseline_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    # for name, value in zip(model.metrics_names, baseline_results):
    #     print(name, ': ', value)
    # test_predictions_baseline = model.predict(X_test, batch_size=5)
    # plot_roc("Test Baseline", y_test, test_predictions_baseline, linestyle='--')
    roc_auc, fpr, tpr, report, acc, p0, p1, r0, r1, f0, f1 = plot_roc(ax, model, X_test, y_test)
    return roc_auc, fpr, tpr, acc, p0, p1, r0, r1, f0, f1


def formatDataForKeras(X):
    loaded = []
    data = []
    for x in X:
        data.append(x)
    loaded.append(np.array(data))
    loaded = dstack(loaded)
    return loaded


def format(trainX, trainy, testX, testy):
    X_train = formatDataForKeras(trainX)
    X_test = formatDataForKeras(testX)
    #binarize target
    trainy = np.array(trainy != 1).astype(int)
    testy = np.array(testy != 1).astype(int)

    y_train = to_categorical(trainy, num_classes=2)
    y_test = to_categorical(testy, num_classes=2)
    return X_train, X_test, y_train, y_test


def formatDataFor2DCnn(X_train, X_test, y_train, y_test):
    train_signals, test_signals = [], []
    # for input_file in os.listdir(train_folder):
    #     # signal = read_signals_ucihar(train_folder + input_file)

    train_signals.append(X_train)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    # for input_file in os.listdir(test_folder):
    #     signal = read_signals_ucihar(test_folder + input_file)
    #     test_signals.append(signal)
    test_signals.append(X_test)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = y_train.tolist()
    test_labels = y_test.tolist()
    return train_signals, train_labels, test_signals, test_labels


def run2DCnn(epochs, cross_validation_method, X, y, class_healthy, class_unhealthy, steps, days, farm_id, sampling, label_series, downsample_false_class, output_dir):
    start_time = time.time()
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    test_balanced_accuracy_score = []
    test_precision_score0 = []
    test_precision_score1 = []
    test_recall_score0 = []
    test_recall_score1 = []
    test_f1_score0 = []
    test_f1_score1 = []
    i = 0
    for train_index, test_index in cross_validation_method.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        roc_auc, fpr, tpr, acc, p0, p1, r0, r1, f0, f1 = evaluate2DCnn(i, output_dir, ax, X_train, y_train, X_test, y_test, epochs=epochs)
        i += 1
        if np.isnan(roc_auc):
            warnings.warn("classifier returned the same target for all testing samples.")
            continue
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        test_balanced_accuracy_score.append(acc)
        test_precision_score0.append(p0)
        test_precision_score1.append(p1)
        test_recall_score0.append(r0)
        test_recall_score1.append(r1)
        test_f1_score0.append(f0)
        test_f1_score1.append(f1)
    clf_detail = "2DCNN"
    mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, output_dir, steps+"_"+clf_detail, fig)
    # fig.show()
    plt.close(fig)
    plt.clf()

    Score(y, class_healthy, class_unhealthy, steps, days, farm_id, test_balanced_accuracy_score,
                 test_precision_score0, test_precision_score1, test_recall_score0, test_recall_score1,
                 test_f1_score0, test_f1_score1, sampling, mean_auc, label_series,
                 cross_validation_method, start_time, output_dir, downsample_false_class, clf_detail=clf_detail)


def evaluate2DCnn(i, out_dir, ax, X_train_, y_train_, X_test_, y_test_, verbose=0, epochs=100, batch_size=8):
    Xtrain, ytrain, Xtest, ytest = formatDataFor2DCnn(X_train_, X_test_, y_train_, y_test_)

    # scales = [2, 10, 20, 30]
    # for k in range(1, 24*7):
    #     scales.append(k*60)
    # scales = np.array(scales)
    #
    # waveletname = 'morl'

    train_size = Xtrain.shape[0]
    test_size = Xtest.shape[0]

    train_data_cwt = None

    for ii in tqdm(range(0, train_size)):
        # print("%d/%d" % (ii, train_size))
        jj = 0
        signal = Xtrain[ii, :, jj]
        # coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        # coeff, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1, wavelet=wavelet.Morlet())
        coeff, _, _, _, scales = cwt_power(signal, out_dir, i=ii, avg=np.average(signal), step_slug="TRAIN", enable_graph_out=False)

        #coeff_2, freq = pywt.cwt(signal, scales, waveletname, 1)
        #coeff, freq = pywt.cwt(y, scales, "db4", 1)

        coeff[np.isnan(coeff)] = 0
        coeff = MinMaxScaler().fit_transform(coeff)
        coeff_ = coeff[:, :X_train_.shape[1]]

        if train_data_cwt is None:
            train_data_cwt = np.ndarray(shape=(train_size, len(scales), Xtrain.shape[1], 1))

        train_data_cwt[ii, :, :, jj] = coeff_

    test_data_cwt = None

    for ii in tqdm(range(0, test_size)):
        # print("%d/%d" % (ii, test_size))
        jj = 0
        signal = Xtest[ii, :, jj]
        coeff, _, _, _, scales = cwt_power(signal, out_dir, i=ii, avg=np.average(signal), step_slug="TEST", enable_graph_out=False)
        ##coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        #coeff, freq = pywt.cwt(y, scales, "db4", 1)
        coeff[np.isnan(coeff)] = 0
        coeff = MinMaxScaler().fit_transform(coeff)
        coeff_ = coeff[:, :X_train_.shape[1]]
        if test_data_cwt is None:
            test_data_cwt = np.ndarray(shape=(test_size, len(scales), X_test_.shape[1], 1))
        test_data_cwt[ii, :, :, jj] = coeff_

    print("calculated (%d) cwt in train fold." % train_size)
    print("calculated (%d) cwt in test fold." % test_size)

    ytrain = (np.array(ytrain) == 1).astype(int)
    ytest = (np.array(ytest) == 1).astype(int)

    x_train = train_data_cwt
    y_train = list(ytrain)
    x_test = test_data_cwt
    y_test = list(ytest)

    input_shape = (train_data_cwt.shape[1], train_data_cwt.shape[2], train_data_cwt.shape[3])

    batch_size = 5
    num_classes = 2
    # epochs = 30

    x_train = x_train.astype('float16')
    x_test = x_test.astype('float16')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 5),
                     activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(10, 10)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
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

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy

    fig_fold, ax_fold = plt.subplots()
    ax_fold.plot(history.history['auc'])
    ##plt.plot(history.history['val_auc'])
    ax_fold.set_title('model auc for fold %d' % i)
    ax_fold.set_ylabel('auc')
    ax_fold.set_xlabel('epoch')
    ax_fold.legend(['train', 'test'], loc='upper left')
    path = "%s/model/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'fold%d_modelauc.png' % (i))
    print(final_path)
    fig_fold.savefig(final_path)
    fig_fold.clear()
    plt.close(fig_fold)

    fig_fold, ax_fold = plt.subplots()
    # summarize history for loss
    ax_fold.plot(history.history['loss'])
    ##plt.plot(history.history['val_loss'])
    ax_fold.set_title('model loss for fold %d' % i)
    ax_fold.set_ylabel('loss')
    ax_fold.set_xlabel('epoch')
    ax_fold.legend(['train', 'test'], loc='upper left')
    path = "%s/model/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'fold%d_modelloss.png' % (i))
    print(final_path)
    fig_fold.savefig(final_path)
    fig_fold.clear()
    plt.close(fig_fold)

    # evaluate model
    baseline_results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    roc_auc, fpr, tpr, report, acc, p0, p1, r0, r1, f0, f1 = plot_roc(ax, model, x_test, y_test)
    return roc_auc, fpr, tpr, acc, p0, p1, r0, r1, f0, f1


def run1DCnn(epochs, cross_validation_method, X, y, class_healthy, class_unhealthy, steps, days, farm_id, sampling, label_series, downsample_false_class, output_dir):
    start_time = time.time()
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    test_balanced_accuracy_score = []
    test_precision_score0 = []
    test_precision_score1 = []
    test_recall_score0 = []
    test_recall_score1 = []
    test_f1_score0 = []
    test_f1_score1 = []
    i = 0
    for train_index, test_index in cross_validation_method.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        roc_auc, fpr, tpr, acc, p0, p1, r0, r1, f0, f1 = evaluate_model(i, output_dir, ax, X_train, y_train, X_test, y_test, epochs=epochs)
        i += 1
        if np.isnan(roc_auc):
            warnings.warn("classifier returned the same target for all testing samples.")
            continue
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        test_balanced_accuracy_score.append(acc)
        test_precision_score0.append(p0)
        test_precision_score1.append(p1)
        test_recall_score0.append(r0)
        test_recall_score1.append(r1)
        test_f1_score0.append(f0)
        test_f1_score1.append(f1)

    mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, output_dir, steps+"_CNN", fig)
    # fig.show()
    plt.close(fig)
    plt.clf()

    Score(y, class_healthy, class_unhealthy, steps, days, farm_id, test_balanced_accuracy_score,
                 test_precision_score0, test_precision_score1, test_recall_score0, test_recall_score1,
                 test_f1_score0, test_f1_score1, sampling, mean_auc, label_series,
                 cross_validation_method, start_time, output_dir, downsample_false_class)


if __name__ == "__main__":
    print("***************************")
    print("CNN")
    print("***************************")
    y = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]).reshape(-1, 1)


    # X = []
    # for _ in range(128):
    #     X.append(np.random.rand(y.size).reshape(-1, 1))
    # X = np.concatenate(X, 1)

    X, y = make_blobs(n_samples=y.size, centers=2, n_features=100, random_state=0, cluster_std=50)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title('centers = 2')
    plt.show()

    animal_ids = np.array(['40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
       '40101310109.0', '40101310109.0', '40101310109.0', '40101310109.0',
       '40101310109.0', '40101310109.0', '40101310013.0', '40101310013.0',
       '40101310013.0', '40101310013.0', '40101310013.0', '40101310013.0',
       '40101310013.0', '40101310013.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310134.0', '40101310134.0', '40101310134.0',
       '40101310134.0', '40101310143.0', '40101310143.0', '40101310143.0',
       '40101310143.0', '40101310143.0', '40101310143.0', '40101310143.0',
       '40101310143.0', '40101310143.0', '40101310143.0', '40101310249.0',
       '40101310249.0', '40101310249.0', '40101310249.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310314.0', '40101310314.0', '40101310314.0', '40101310314.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310316.0', '40101310316.0', '40101310316.0',
       '40101310316.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310342.0', '40101310342.0', '40101310342.0',
       '40101310342.0', '40101310350.0', '40101310350.0', '40101310350.0',
       '40101310350.0', '40101310350.0', '40101310350.0', '40101310350.0',
       '40101310350.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310353.0', '40101310353.0', '40101310353.0',
       '40101310353.0', '40101310386.0', '40101310386.0', '40101310386.0',
       '40101310386.0', '40101310386.0', '40101310386.0', '40101310386.0',
       '40101310386.0', '40101310386.0', '40101310069.0', '40101310069.0',
       '40101310069.0', '40101310069.0', '40101310069.0', '40101310069.0',
       '40101310069.0', '40101310069.0', '40101310098.0', '40101310098.0',
       '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0',
       '40101310098.0', '40101310098.0', '40101310098.0', '40101310098.0'])

    sample_idx = [1, 2, 3, 6, 8, 9, 10, 13, 14, 17, 18, 19, 21, 22, 24, 28, 30, 34, 35, 36, 38, 39, 43, 45, 46, 47, 50,
                  51, 52, 55, 56, 59, 60, 61, 62, 63, 65, 67, 69, 71, 73, 76, 80, 81, 82, 85, 87, 88, 89, 90, 91, 92,
                  94, 96, 97, 98, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                  124, 127, 128, 130, 132, 133, 134, 135, 137, 138, 143, 145, 146, 147, 149, 153, 155, 156, 158, 160,
                  162, 163, 165, 167, 169, 170, 171, 172, 174, 175, 177, 178, 179, 180, 181, 183, 185, 188, 189, 191,
                  194, 197, 198, 199, 201, 205, 206, 207, 209, 210, 211, 214, 215, 219, 220]

    print("DATASET:")
    dataset = pd.DataFrame(np.hstack((X, y.reshape(y.size, 1), animal_ids.reshape(animal_ids.size, 1))))
    dataset.to_csv("dummy_dataset_for_cv.csv")
    print(dataset)
    print("")
    output_dir = "F:/Data2/cnn_debug"
    label_series = {0: "l0", 1: "l1", 2: "l2", 3: "l3", 4: "l4"}
    stratified = False
    cross_validation_method = StratifiedLeaveTwoOut(animal_ids, sample_idx, stratified=stratified, verbose=True)
    run1DCnn(cross_validation_method, X, y, animal_ids, sample_idx, 1, 2, "steps->2", 7, "farm_id", "sampling", label_series, False, output_dir)