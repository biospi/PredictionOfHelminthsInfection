import warnings
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
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

from utils._custom_split import StratifiedLeaveTwoOut
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc
from sklearn.preprocessing import binarize
from utils.visualisation import mean_confidence_interval, plot_roc_range
from keras.utils.vis_utils import plot_model
import time
import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'


class Score:
    def __init__(self, y, class_healthy, class_unhealthy, steps, days, farm_id, test_balanced_accuracy_score,
                 test_precision_score0, test_precision_score1, test_recall_score0, test_recall_score1,
                 test_f1_score0, test_f1_score1, sampling, mean_auc, label_series,
                 cross_validation_method, start_time, output_dir, downsample_false_class):
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
        scores["classifier_details"] = "1DCNN"
        scores["roc_auc_score_mean"] = mean_auc
        report_rows_list.append(scores)

        df_report = pd.DataFrame(report_rows_list)
        df_report["class_0_label"] = label_series[class_healthy]
        df_report["class_1_label"] = label_series[class_unhealthy]
        df_report["nfold"] = cross_validation_method.nfold if hasattr(cross_validation_method, 'nfold') else np.nan
        df_report["total_fit_time"] = [time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))]
        filename = "%s/%s_classification_report_days_%d_option_%s_downsampled_%s_sampling_%s.csv" % (
            output_dir, farm_id, days, steps, downsample_false_class, sampling)
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


def evaluate_model(ax, trainX, trainy, testX, testy, verbose=0, epochs=1000, batch_size=8):
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
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
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
    for train_index, test_index in cross_validation_method.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        roc_auc, fpr, tpr, acc, p0, p1, r0, r1, f0, f1 = evaluate_model(ax, X_train, y_train, X_test, y_test, epochs=epochs)
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