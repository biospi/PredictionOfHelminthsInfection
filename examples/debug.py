import os

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, f1_score, roc_curve
import numpy as np
from sys import exit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go

from cwt._cwt import compute_spectogram_matlab
from utils.visualisation import plotMlReport, plotMlReportFinal


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
            label='Chance', alpha=1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
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
    fig.show()
    return mean_auc


def make_roc_curve(out_dir, classifier, X, y, cv, param_str):
    print("make_roc_curve")

    if isinstance(X, pd.DataFrame):
        X = X.values
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.clf()
    fig, ax = plt.subplots()
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
    # clf_name = "%s_%s" % ("_".join([x[0] for x in classifier.steps]), param_str)

    mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, "clf_name", fig)

    plt.close(fig)
    plt.clf()
    aucs2 = np.mean(aucs)
    return mean_auc, aucs2

from plotnine import *
import glob
from pathlib import Path
import json
import re
from plotly.subplots import make_subplots
from shutil import copyfile


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def format(text):
    return text.replace("activity_no_norm", "TimeDom->")\
        .replace("activity_quotient_norm", "TimeDom->QN->")\
        .replace("cwt_quotient_norm", "TimeDom->QN->CWT->")\
        .replace("cwt_no_norm", "TimeDom->CWT->") \
        .replace("cwt_quotient_no_norm", "TimeDom->CWT->") \
        .replace("humidity", "Humidity->") \
        .replace("_humidity", "Humidity->") \
        .replace(",", "").replace("(", "").replace(")", "").replace("'","").replace(" ","").replace("->->","->").replace("_","->")


if __name__ == "__main__":
    # clf = SVC(kernel="linear", probability=True)
    # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    # y = np.array([1, 1, 2, 2])
    # clf.fit(X, y)
    #
    # gamma = clf._gamma
    # coef0 = clf.coef0
    # cost = clf.C
    # tolerance = clf.tol
    # probability_ = clf.probability
    #
    # exit()
    # scales = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10080])
    #
    # scales = [np.power(2, n) for n in np.arange(2, 10)]
    #
    # import pycwt as wavelet
    # # f0s = [6,10]
    # # fig, axs = plt.subplots(len(f0s), 1, figsize=(7.20, 12.80))
    # # for i, f0 in enumerate(f0s):
    # #     wavelet_func = [wavelet.Morlet(f0).psi(x) for x in np.arange(-10, 10, 0.1)]
    # #     axs[i].plot(wavelet_func)
    # #     axs[i].title.set_text("Morlet wavelet with wave number= %s" % str(f0))
    # # fig.show()
    #
    # f0 = 6
    # fig, ax = plt.subplots(figsize=(7.20, 7.20))
    # wavelet_func = [wavelet.MexicanHat().psi(x) for x in np.arange(-10, 10, 0.1)]
    # ax.plot(wavelet_func)
    # ax.title.set_text("Mexican hat wavelet")
    # fig.show()
    #
    # f0 = 6
    # fig, ax = plt.subplots(figsize=(7.20, 7.20))
    # wavelet_func = [wavelet.MexicanHat().psi(x) for x in np.arange(-10, 10, 0.1)]
    # ax.plot(wavelet_func)
    # ax.title.set_text("Morlet wavelet with wave number= %s" % str(f0))
    # fig.show()
    # exit()

    # path = "F:/Data2/job_cedara_debug/ml/ml_kfold_2to3_3day/RepeatedKFold/final_classification_report.csv"
    # output_dir = "/".join(path.split("/")[:-1])
    # plotMlReport(path, output_dir)
    # exit()
    directory = "F:/Data2/job_cedara_debug_night/ml"
    output_dir = "/".join(directory.split("/")[:-1])
    all_csv_files = [file for path, subdir, files in os.walk(directory) for file in glob.glob(os.path.join(path, "*.csv"))]
    all_csv_files = [x for x in all_csv_files if "final" in x]
    plotMlReportFinal(all_csv_files, output_dir)
    exit()

    # # import the libraries
    # from scipy import signal
    # import matplotlib.pyplot as plot
    # import numpy as np
    #
    # # Define the list of frequencies
    # frequencies = np.arange(5, 105, 5)
    # # Sampling Frequency
    # samplingFrequency = 400
    # # Create two ndarrays
    # s1 = np.empty([0])  # For samples
    # s2 = np.empty([0])  # For signal
    # # Start Value of the sample
    # start = 1
    # # Stop Value of the sample
    # stop = samplingFrequency + 1
    # for frequency in frequencies:
    #     sub1 = np.arange(start, stop, 1)
    #     # Signal - Sine wave with varying frequency + Noise
    #     sub2 = np.sin(2 * np.pi * sub1 * frequency * 1 / samplingFrequency) + np.random.randn(len(sub1))
    #     s1 = np.append(s1, sub1)
    #     s2 = np.append(s2, sub2)
    #     start = stop + 1
    #     stop = start + samplingFrequency
    # # Plot the signal
    # plot.subplot(211)
    # plot.plot(s1, s2)
    # plt.show()
    #
    # freqs, _, coefs = signal.stft(s2, fs=1, nperseg=256)
    #
    # # import matlab.engine
    # # matlab = matlab.engine.start_matlab()
    # # matlab=None
    # # t = np.linspace(0, 5, 5000)
    # # x = 2 * np.cos(2 * np.pi * 100 * t) * (t < 1) + np.cos(2 * np.pi * 50 * t) * (3 < t) + 0.3 * np.random.rand(np.size(t))
    # # plt.plot(x)
    # # plt.show()
    # # scales = np.arange(1, 150)
    # # coefs, _, scales, freqs = compute_spectogram_matlab(x, scales)
    # #
    # # # Fs = matlab.double(1/60)
    # # # scales = np.arange(1, 150)
    # # # mat_scales = matlab.double(matlab.cell2mat(scales.astype(np.float).tolist()))
    # # # mat_a = matlab.double(matlab.cell2mat(x.tolist()))
    # # # #wt, freqs = matlab.cwt(mat_a, mat_scales, 'sym4', Fs, nargout=2)
    # # # wt, freqs, _ = matlab.spectrogram(mat_a, nargout=3)
    # #
    # #coefs = np.asarray(wt)
    # coefs_cc = np.conj(coefs)
    # power = np.real(np.multiply(coefs, coefs_cc))
    #
    # fig, ax = plt.subplots()
    # pos = ax.imshow(power, extent=[0, power.shape[1], 0, power.shape[0]])
    # fig.colorbar(pos, ax=ax)
    # #axs[1].plot(coi_line_array, linestyle="--", linewidth=1, c="red")  # todo fix xratio
    # ax.set_aspect('auto')
    # #ax.set_title("CWT")
    # ax.set_xlabel("Time in minute")
    # #ax.set_ylabel("Wave length of wavelet (in minute)")
    # fig.show()
    #
    # exit()
    #
    #
    # from scipy import signal
    # # from ssqueezepy import ssq_cwt, ssq_stft
    # #
    # # Twxo, Wxo, *_ = ssq_cwt(xo)
    # import pycwt as wavelet
    # import pywt
    # [r, psi, x] = pywt.Wavelet('db4').wavefun()
    # plt.plot(x, psi)
    # plt.show()
    #
    # scales = [1, 5, 10, 20, 30, 40, 50, 100]
    # y = list(range(100))
    # #coeff, freq = pywt.dwt2(y, scales, "db4", 1)
    # #cwtmatr = signal.cwt(y, signal.daub(4), scales)
    #


    # import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # from sklearn import decomposition
    # from sklearn import datasets
    #
    # np.random.seed(5)
    #
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    #
    # fig = plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #
    # plt.cla()
    # pca = decomposition.PCA(n_components=3)
    # pca.fit(X)
    # X = pca.transform(X)
    #
    # for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    #     ax.text3D(X[y == label, 0].mean(),
    #               X[y == label, 1].mean() + 1.5,
    #               X[y == label, 2].mean(), name,
    #               horizontalalignment='center',
    #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(float)
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
    #            edgecolor='k')
    #
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    #
    # plt.show()

    # path = "F:/Data2/biospi_last/ml_gain_1_4_7day/final_classification_report.csv"
    # output_dir = "/".join(path.split("/")[:-1])
    # plotMlReport(path, output_dir)

    # df = pd.read_csv(str(path), index_col=None)
    # df["config"] = [format(str(x)) for x in list(zip(df.steps, df.classifier))]
    # df = df.sort_values('roc_auc_score_mean')
    # df = df.drop_duplicates(subset=['config'], keep='first')
    #
    # print(df)
    #
    # t4 = "AUC performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
    # df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
    # df["class_1_label"].values[0])
    #
    # t3 = "Accuracy performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
    # df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
    # df["class_1_label"].values[0])
    #
    # t1 = "Precision class0 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
    # df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
    # df["class_1_label"].values[0])
    #
    # t2 = "Precision class1 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
    # df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
    # df["class_1_label"].values[0])
    #
    # fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))
    #
    # fig.append_trace(px.bar(df, x='config', y='precision_score0_mean', range_y=[df["precision_score0_mean"].min(), df["precision_score0_mean"].max]).data[0], row=1, col=1)
    # fig.append_trace(px.bar(df, x='config', y='precision_score1_mean').data[0], row=2, col=1)
    # fig.append_trace(px.bar(df, x='config', y='balanced_accuracy_score_mean').data[0], row=3, col=1)
    # fig.append_trace(px.bar(df, x='config', y='roc_auc_score_mean').data[0], row=4, col=1)
    #
    #
    #
    # fig.update_yaxes(range=[0, 1], row=1, col=1)
    # fig.update_yaxes(range=[0, 1], row=2, col=1)
    # fig.update_yaxes(range=[0, 1], row=3, col=1)
    # fig.update_yaxes(range=[0, 1], row=4, col=1)
    #
    # fig.add_shape(type="line", x0=-0.0, y0=0.920, x1=1.0, y1=0.920, line=dict(color="LightSeaGreen", width=4, dash="dot",))
    #
    # fig.add_shape(type="line", x0=-0.0, y0=0.640, x1=1.0, y1=0.640,
    #               line=dict(color="LightSeaGreen", width=4, dash="dot", ))
    #
    # fig.add_shape(type="line", x0=-0.0, y0=0.357, x1=1.0, y1=0.357,
    #               line=dict(color="LightSeaGreen", width=4, dash="dot", ))
    #
    # fig.add_shape(type="line", x0=-0.0, y0=0.078, x1=1.0, y1=0.078,
    #               line=dict(color="LightSeaGreen", width=4, dash="dot", ))
    #
    # fig.update_xaxes(showticklabels=False)  # hide all the xticks
    # fig.update_xaxes(showticklabels=True, row=4, col=1)
    #
    # # fig.update_layout(shapes=[
    # #     dict(
    # #         type='line',
    # #         color="MediumPurple",
    # #         yref='paper', y0=0.945, y1=0.945,
    # #         xref='x', x0=-0.5, x1=7.5
    # #     )
    # # ])
    # fig.update_yaxes(showgrid=True, gridwidth=1)
    # fig.update_xaxes(showgrid=True, gridwidth=1)
    # fig.write_html("ML_performance.html")
    # fig.show()
    #
    # # files = []
    # dfs = []
    # for path in Path("F:\\Data2\\biospi").rglob('*.csv'):
    #     if 'final' not in path.name:
    #         continue
    #     print(path)
    #     df = pd.read_csv(str(path), index_col=None)
    #     df["dataset"] = str(path).split('\\')[-2]
    #     print(df)
    #     dfs.append(df)
    #
    # df_merged = pd.concat(dfs)
    # df_merged.to_csv("F:\\Data2\\biospi\\result_merge.csv", index=False)


    # files = []
    # for path in Path("C:\\Users\\fo18103\\OneDrive - University of Bristol\\South Africa\\backfill_1min_xyz_delmas_fixed").rglob('*.csv'):
    #     if int(path.name.split('.')[0]) not in [40101310316, 40101310040, 40101310109, 40101310110, 40101310353, 40101310314, 40101310085, 40101310143, 40101310409, 40101310134, 40101310342, 40101310069, 40101310013, 40101310098, 40101310350, 40101310386, 40101310249]:
    #         continue
    #     files.append(str(path))
    #     print(str(path))
    #     dst = str(path).replace("backfill_1min_xyz_delmas_fixed", "top_17")
    #     copyfile(str(path), dst)
    #
    # exit()
    # iterations = list(range(1000))
    # for i in iterations:
    #     if (i % 100 == 0) | (i == 0) | (i == iterations[-1]):
    #         print(i)
    # exit()



    # DIR = "F:/Data2/imp_full_reshape_xyz_debug_59"
    #
    # rmse_list = []
    # rmse_list_li = []
    # files = []
    # for path in Path(DIR).rglob('*.json'):
    #     files.append(str(path))
    #
    # files.sort(key=natural_keys)
    #
    # for path in files:
    #     print(path)
    #     with open(path) as json_file:
    #         data = json.load(json_file)
    #         print(data["rmse"])
    #         rmse_list.append(data["rmse"])
    #         rmse_list_li.append(data["rmse_li"])
    #
    # plt.clf()
    # plt.cla()
    # fig, ax = plt.subplots()
    # ax.set_ylabel('RMSE')
    # ax.set_xlabel('iteration')
    # plt.plot([x * 10 for x in range(len(rmse_list))], rmse_list, label="RMSE GAIN", alpha=1)
    # plt.plot([x * 10 for x in range(len(rmse_list_li))], rmse_list_li, label="RMSE LI", alpha=1)
    #
    # plt.title("RMSE iteration performance")
    # plt.legend()
    # plt.show()
    #
    #
    # exit()
    #
    #
    # df = pd.read_csv("z_prct_data.csv")
    # g = (ggplot(df)  # defining what data to use
    #  + aes(x='Target', y='Percent of zeros', color='Target', shape='Target')  # defning what variable to usei
    #  + geom_jitter()  # defining the type of plot to use
    #  + stat_summary(geom="crossbar", color="black", width=0.2)
    #  + theme(subplots_adjust={'right': 0.82})
    #  )
    #
    # fig = g.draw()
    # fig.tight_layout()
    # fig.show()
    #
    # exit(0)
    #
    #
    #
    # # generate 2d classification dataset
    # X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    # # scatter plot, dots colored by class value
    # df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    # colors = {0: 'red', 1: 'blue', 2: 'green'}
    # fig, ax = plt.subplots()
    # grouped = df.groupby('label')
    # for key, group in grouped:
    #     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    #
    # scoring = {
    #     'balanced_accuracy_score': make_scorer(balanced_accuracy_score)
    # }
    #
    # clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
    # cv_svc = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,
    #                                  random_state=0)
    # scores = cross_validate(clf_svc, X.copy(), y.copy(), cv=cv_svc, n_jobs=-1, scoring=scoring, return_estimator=True)
    #
    #
    # # scores["roc_auc_score_mean"] = np.mean(scores["test_roc_auc_score"])
    # a, b = make_roc_curve("out_dir", clf_svc, X.copy(), y.copy(), cv_svc, "param_str")
    # # auc_cv = []
    # # for e in scores['estimator']:
    # #     y_pred = e.predict_proba(X)
    # #     y_true = y
    # #     aucs = []
    # #     for fold in range(y_pred.shape[1]):
    # #         y_p = y_pred[:, fold]
    # #         fpr, tpr, _ = metrics.roc_curve(y_true, y_p, pos_label=1)
    # #         aucs.append(metrics.auc(fpr, tpr))
    # #     auc_cv.append(np.mean(aucs))
    # # auc_cv_mean = np.mean(auc_cv)
    # exit(0)
    #
    #
    # classifier = make_pipeline(SVC(probability=True, class_weight='balanced'))
    # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10,
    #                                  random_state=0)
    #
    #
    # tprs = []
    # fprs = []
    # aucs = []
    # bas = []
    # for i, (train, test) in enumerate(cv.split(X, y)):
    #     classifier.fit(X[train], y[train])
    #     # viz = plot_roc_curve(classifier, X[test], y[test],
    #     #                      label=None,
    #     #                      alpha=0.3, lw=1, ax=ax, c="tab:blue")
    #     # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    #     # interp_tpr[0] = 0.0
    #     # tprs.append(interp_tpr)
    #     # aucs.append(viz.roc_auc)
    #     # ax.plot(viz.fpr, viz.tpr, c="tab:green")
    #     y_true = y[test]
    #     y_pred = classifier.predict(X[test])
    #
    #     fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    #     tprs.append(tpr)
    #     fprs.append(fpr)
    #     aucs.append(metrics.auc(fpr, tpr))
    #     bas.append(balanced_accuracy_score(y_true, y_pred))
    #     p_s = precision_score(y_true, y_pred, average=None)
    #     p_s_0 = p_s[0]
    #     p_s_1 = p_s[1]
    #
    #     r_s = recall_score(y_true, y_pred, average=None)
    #     r_s_0 = r_s[0]
    #     r_s_1 = r_s[1]
    #


