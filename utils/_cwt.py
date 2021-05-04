import os

import matplotlib
from matplotlib.ticker import ScalarFormatter
from sklearn.utils import check_array
from sklearn.base import TransformerMixin, BaseEstimator
import pycwt as wavelet
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sys import exit
import random
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
import plotly.express as px

from utils.Utils import create_rec_dir, anscombe
from utils.visualisation import plot_cwt_power_sidebyside


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

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic iteration")
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/png/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)

    path = "%s/roc_curve/svg/" % out_dir
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    print(final_path)
    fig.savefig(final_path)
    return mean_auc


def make_roc_curve(out_dir, classifier, X, y, cv, param_str):
    print("make_roc_curve")
    details = str(classifier).replace('\n', '').replace(" ", '')
    slug = "".join(x for x in details if x.isalnum())

    clf_name = "%s_%s" % (slug, param_str)
    print(clf_name)

    if isinstance(X, pd.DataFrame):
        X = X.values

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.clf()
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        if isinstance(cv, RepeatedStratifiedKFold):
            print("make_roc_curve fold %d/%d" % (i, cv.get_n_splits()))
        else:
            print("make_roc_curve fold %d/%d" % (i, cv.nfold))
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             label=None,
                             alpha=0.3, lw=1, ax=ax, c="tab:blue")
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        if np.isnan(viz.roc_auc):
            continue
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        # ax.plot(viz.fpr, viz.tpr, c="tab:green")
    print("make_roc_curve done!")
    mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, clf_name, fig)
    plt.close(fig)
    plt.clf()
    return mean_auc


def plot_cwt_power(step_slug, out_dir, i, activity, power_masked, coi_line_array, freqs, format_xaxis=True):
    wavelength = 1/freqs
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(19.20, 7.20))
    fig.suptitle("Signal , CWT", fontsize=18)
    ticks = list(range(len(activity)))
    if format_xaxis:
        ticks = get_time_ticks(len(activity))
    axs[0].plot(ticks, activity)
    axs[0].set_title("Time domain signal")

    axs[0].set(xlabel="Time in minute", ylabel="activity")
    if format_xaxis:
        axs[0].set(xlabel="Time", ylabel="activity")

    if format_xaxis:
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[0].xaxis.set_major_locator(mdates.DayLocator())

    with np.errstate(invalid='ignore'):  # ignore numpy divide by zero warning
        if step_slug == "QN_CWT_ANSCOMBE_LOG":
            power_masked = np.log(anscombe(power_masked))
        axs[1].imshow(power_masked)
    if(len(coi_line_array) > 0):
        axs[1].plot(coi_line_array, linestyle="--", linewidth=0, c="white")#todo fix xratio
    axs[1].set_aspect('auto')
    axs[1].set_title("CWT")
    axs[1].set_xlabel("Time in minute")
    if format_xaxis:
        axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Wave length of wavelet (in minute)")
    #axs[1].set_yscale('log')

    if format_xaxis:
        n_x_ticks = axs[1].get_xticks().shape[0]
        labels_ = [item.strftime("%H:00") for item in ticks]
        labels_ = np.array(labels_)[list(range(1, len(labels_), int(len(labels_) / n_x_ticks)))]
        labels_[:] = labels_[0]
        axs[1].set_xticklabels(labels_)

    # n_y_ticks = axs[1].get_yticks().shape[0]-2
    cwty = axs[1].get_yticks()
    n_y_ticks = cwty.shape[0] - len([x for x in cwty if x < 0])

    labels_wl = ["%.2f" % item for item in wavelength]
    # print(labels)
    labels_wt = np.array(labels_wl)[list(range(1, len(labels_wl), int(len(labels_wl) / n_y_ticks)))][1:]
    #new_lab = [matplotlib.text.Text(0, float(labels_wt[0]), labels_wt[0])]
    new_lab = []
    for ii, l in enumerate(labels_wt):
        new_lab.append(matplotlib.text.Text(cwty[ii], float(l), l))
    #new_lab[-1] = matplotlib.text.Text(8, float(l), l)
    axs[1].set_yticklabels(new_lab)

    axs[1].tick_params(axis='y', which='both', colors='black')

    filename = "%d_%s_cwt.png" % (i, step_slug)
    filepath = "%s/%s" % (out_dir, filename)
    create_rec_dir(filepath)
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)


def mask_cwt(cwt, coi, scales, turn_off=False):
    if turn_off:
        return cwt
    # print("masking cwt...")

    coi_line = []
    for j in range(cwt.shape[1]):
        for i, s in enumerate(scales):
            c = coi[j]
            if s > c:
                cwt[i:, j] = np.nan
                coi_line.append(i)
                break

    return cwt, coi_line


def CWTVisualisation(step_slug, graph_outputdir, shape, freqs, coi_line_array,
                  df_timedomain, df_cwt,
                  class_healthy_label, class_unhealthy_label,
                  class_healthy, class_unhealthy):
    idx_healthy = df_timedomain[df_timedomain["target"] == class_healthy].index.tolist()
    idx_unhealthy = df_timedomain[df_timedomain["target"] == class_unhealthy].index.tolist()
    h_m = np.mean(df_cwt.loc[idx_healthy].values, axis=0).reshape(shape)
    uh_m = np.mean(df_cwt.loc[idx_unhealthy].values, axis=0).reshape(shape)
    plot_cwt_power_sidebyside(step_slug, True, class_healthy_label, class_unhealthy_label, class_healthy,
                                  class_unhealthy, idx_healthy, idx_unhealthy, coi_line_array, df_timedomain,
                                  graph_outputdir, h_m, uh_m, freqs, ntraces=2)


def check_scale_spacing(scales):
    spaces = []
    for i in range(scales.shape[0] - 1):
        wavelet_scale_space = scales[i] - scales[i + 1]
        spaces.append(wavelet_scale_space)
    print(spaces)
    print(np.min(spaces), np.max(spaces))
    return np.mean(spaces)


def cwt_power(activity, out_dir, i=0, step_slug="CWT_POWER", format_xaxis=None):
    y = activity
    wavelenght = len(activity) #nday wavelenght in minutes
    f0 = 1 / wavelenght
    w = wavelet.Morlet(f0)
    #w = wavelet.MexicanHat()
    coefs, scales, freqs, coi, _, _ = wavelet.cwt(y, 0.1, wavelet=w, dj=1./30.)
    #mean_scale_space = check_scale_spacing(scales)
    coefs_cc = np.conj(coefs)
    with np.errstate(divide='ignore'):  # ignore numpy divide by zero warning
        # power_cwt = np.log(np.real(np.multiply(coefs, coefs_cc)))
        power_cwt = np.real(np.multiply(coefs, coefs_cc))
    # power_cwt[power_cwt == -np.inf] = 0  # todo check why inf output
    power_masked, coi_line_array = mask_cwt(power_cwt.copy(), coi, scales)
    shape = power_cwt.shape
    # power_masked, coi_line_array = power_cwt, []
    plot_cwt_power(step_slug, out_dir, i, activity, power_masked.copy(), coi_line_array, freqs, format_xaxis=format_xaxis)
    return power_masked, freqs, coi, shape


def compute_cwt(X, out_dir, step_slug, format_xaxis=None):
    print("compute_cwt...")
    out_dir = out_dir + "_cwt"
    plotHeatmap(X, out_dir=out_dir, title="Time domain samples", force_xrange=True, filename="time_domain_samples.html")
    cwt = []
    cwt_full = []
    i = 0
    for activity in tqdm(X):
        power_masked, freqs, coi, shape = cwt_power(activity, out_dir, i, step_slug, format_xaxis)
        power_flatten_masked = np.array(power_masked.flatten())
        cwt_full.append(power_flatten_masked)
        power_flatten_masked = power_flatten_masked[~np.isnan(power_flatten_masked)]#remove masked values
        cwt.append(power_flatten_masked)
        i += 1
    cwt = np.array(cwt)
    cwt_full = np.array(cwt_full)

    # plotHeatmap(cwt, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT.html", head=False)
    plotHeatmap(cwt, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT_sub.html", head=True)
    return cwt, cwt_full, freqs, coi, shape


class CWT(TransformerMixin, BaseEstimator):
    def __init__(self, *, out_dir=None, copy=True, step_slug=None, format_xaxis=False):
        self.out_dir = out_dir
        self.copy = copy
        self.freqs = None
        self.coi = None
        self.shape = None
        self.step_slug = step_slug
        self.format_xaxis = format_xaxis

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse='csr')
        return self

    def transform(self, X, copy=None):
        #copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr')
        cwt, cwt_full, freqs, coi, shape = compute_cwt(X, self.out_dir, self.step_slug, self.format_xaxis)
        self.freqs = freqs
        self.coi = coi
        self.shape = shape
        self.cwt_full = cwt_full
        return cwt


def createSynthetic(activity):
    pure = activity
    noise = np.random.normal(0, 200, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 1.5)
    synt[synt < 0] = 0
    return synt.astype(int)


def createSyntheticActivityData(n_samples=4):
    print("createSyntheticActivityData")
    samples_path = "F:/Data2/dataset_gain_7day/activity_delmas_70101200027_dbft_7_1min.csv"
    df = pd.read_csv(samples_path, header=None)
    df = df.fillna(0)
    crop = -4 - int(df.shape[1]/1.1)
    activity = df.iloc[259, 9353: 9353+60*6].values

    dataset = []
    for j in range(n_samples):
        A = createSynthetic(activity)
        dataset.append(A)

    return dataset


def plotLine(X, out_dir="", title="title", filename="file.html"):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = get_time_ticks(len(sample))
        trace = go.Line(
            opacity=.8,
            x=timestamp,
            y=sample,
        )
        fig.append_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    # fig.show()
    return trace, title


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def plotHeatmap(X, out_dir="", title="Heatmap", filename="heatmap.html", force_xrange=False, head=False):
    # fig = make_subplots(rows=len(transponders), cols=1)
    if head and X.shape[0] > 4:
        X = X[[0, 1, -2, -1]]
    ticks = get_time_ticks(X.shape[1])
    if force_xrange:
        ticks = list(range(X.shape[1]))

    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(
            z=X,
            x=ticks,
            y=list(range(X.shape[0])),
            colorscale='Viridis')
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    # fig.show()
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


def createSinWave(f, time):
    t = np.linspace(0, time, int(time))
    y = np.sin(2. * np.pi * t * f)*100
    return y.astype(float)


def createPoisson(time, s):
    seed = np.ceil(random.random()*2)+1
    noise = np.random.poisson(int(seed), size=int(time))
    y = noise*s
    return y


def createNormal(f, time, s):
    seed = np.ceil(f * 2) + 1
    noise = np.random.normal(seed, size=int(time))
    y = noise*s
    return y


def createPoisonWaves(d, signal10, signal2):
    targets = []
    waves = []
    t = d
    for s in signal10:
        waves.append(createPoisson(t, s))
        targets.append(1)
    for s in signal2:
        waves.append(createPoisson(t, s))
        targets.append(2)
    waves = np.array(waves)
    return waves, targets


def createNormalWaves(d, signal10, signal2):
    targets = []
    waves = []
    t = d
    for s in signal10:
        waves.append(createNormal(random.random(), t, s))
        targets.append(1)
    for s in signal2:
        waves.append(createNormal(random.random(), t, s))
        targets.append(2)
    waves = np.array(waves)
    return waves, targets


def creatSin(freq, time):
    t = np.linspace(0, time, int(time))
    y = np.sin(2. * np.pi * t * freq)
    y[y < 0] = 0
    return y


if __name__ == "__main__":
    print("********CWT*********")

    X = []
    for i in np.arange(5, 100, 5):
        X.append(creatSin(i, 1440))
    X = np.array(X)
    X_CWT = CWT(out_dir="F:/Data2/_cwt_unit", format_xaxis=False).transform(X)

    for d in [(60*60*24*1)/60, (60*60*24*7)/60]:

        signal10 = []
        for _ in range(60):
            signal10.append(creatSin(15 + np.random.random() / 100, d))
        signal2 = []
        for _ in range(60):
            signal2.append(creatSin(2 + np.random.random() / 100, d))

        for out_dir in ["F:/Data2/_cwt_debug_poisson_%d/" % d, "F:/Data2/_cwt_debug_normal_%d/" % d]:
            #X = np.array(createSyntheticActivityData()
            #X, targets = createSinWaves(d)
            #X, targets = createPoisonWaves(d)
            if "poisson" in out_dir:
                X, targets = createPoisonWaves(d, signal10, signal2)

            if "normal" in out_dir:
                X, targets = createNormalWaves(d, signal10, signal2)

            #X = pd.concat([pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X)], axis=1)
            #X.columns = list(range(X.shape[1]))
            #plotLine(X, out_dir=out_dir, title="Activity samples", filename="X.html")

            X_CWT = CWT(out_dir=out_dir, format_xaxis=False).transform(X)

            #plotLine(X_CWT, out_dir=out_dir, title="CWT samples", filename="CWT.html")
            print("********************")
            report_rows_list = []
            y = np.array(targets)
            y = y.astype(int)

            class_healthy = 1
            class_unhealthy = 2
            cross_validation_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
            scoring = {
                'balanced_accuracy_score': make_scorer(balanced_accuracy_score),
                # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
                'precision_score0': make_scorer(precision_score, average=None, labels=[class_healthy]),
                'precision_score1': make_scorer(precision_score, average=None, labels=[class_unhealthy]),
                'recall_score0': make_scorer(recall_score, average=None, labels=[class_healthy]),
                'recall_score1': make_scorer(recall_score, average=None, labels=[class_unhealthy]),
                'f1_score0': make_scorer(f1_score, average=None, labels=[class_healthy]),
                'f1_score1': make_scorer(f1_score, average=None, labels=[class_unhealthy])
            }

            print('TimeDom->SVC')
            clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
            scores = cross_validate(clf_svc, X.copy(), y.copy(), cv=cross_validation_method, scoring=scoring, n_jobs=-1)
            scores["class0"] = y[y == class_healthy].size
            scores["class1"] = y[y == class_unhealthy].size
            scores["option"] = "TimeDom"
            scores["days"] = d
            scores["farm_id"] = 0
            scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
            scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
            scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
            scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
            scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
            scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
            scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
            scores["classifier"] = "->SVC"
            scores["classifier_details"] = str(clf_svc).replace('\n', '').replace(" ", '')
            clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
            aucs = make_roc_curve(out_dir, clf_svc, X.copy(), y.copy(), cross_validation_method, "time_dom")
            scores["roc_auc_score_mean"] = aucs
            print(aucs)
            report_rows_list.append(scores)
            del scores

            print('CWT->SVC')
            clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
            scores = cross_validate(clf_svc, X_CWT.copy(), y.copy(), cv=cross_validation_method, scoring=scoring, n_jobs=-1)
            scores["class0"] = y[y == class_healthy].size
            scores["class1"] = y[y == class_unhealthy].size
            scores["option"] = "CWT"
            scores["days"] = d
            scores["farm_id"] = 0
            scores["balanced_accuracy_score_mean"] = np.mean(scores["test_balanced_accuracy_score"])
            scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
            scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
            scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
            scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
            scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
            scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
            scores["classifier"] = "->SVC"
            scores["classifier_details"] = str(clf_svc).replace('\n', '').replace(" ", '')
            clf_svc = make_pipeline(SVC(probability=True, class_weight='balanced'))
            aucs = make_roc_curve(out_dir, clf_svc, X_CWT.copy(), y.copy(), cross_validation_method, "freq_dom")
            scores["roc_auc_score_mean"] = aucs
            print(aucs)
            report_rows_list.append(scores)
            del scores

            df_report = pd.DataFrame(report_rows_list)

            if not os.path.exists(out_dir):
                print("mkdir", out_dir)
                os.makedirs(out_dir)
            filename = "%s/report.csv" % (out_dir)
            df_report.to_csv(filename, sep=',', index=False)
            print("filename=", filename)

            print("REPORT")
            df = pd.read_csv(str(filename), index_col=None)
            df["class_0_label"] = "1"
            df["class_1_label"] = "2"
            df["config"] = [format(str(x)) for x in list(zip(df.option, df.classifier))]
            df = df.sort_values('roc_auc_score_mean')

            print(df)

            t4 = "AUC performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
            df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
            df["class_1_label"].values[0])

            t3 = "Accuracy performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
            df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
            df["class_1_label"].values[0])

            t1 = "Precision class0 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
            df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
            df["class_1_label"].values[0])

            t2 = "Precision class1 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
            df["days"].values[0], df["class0"].values[0], df["class_0_label"].values[0], df["class1"].values[0],
            df["class_1_label"].values[0])

            fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))

            fig.append_trace(px.bar(df, x='config', y='precision_score0_mean').data[0], row=1, col=1)
            fig.append_trace(px.bar(df, x='config', y='precision_score1_mean').data[0], row=2, col=1)
            fig.append_trace(px.bar(df, x='config', y='balanced_accuracy_score_mean').data[0], row=3, col=1)
            fig.append_trace(px.bar(df, x='config', y='roc_auc_score_mean').data[0], row=4, col=1)

            fig.update_yaxes(range=[0, 1], row=1, col=1)
            fig.update_yaxes(range=[0, 1], row=2, col=1)
            fig.update_yaxes(range=[0, 1], row=3, col=1)
            fig.update_yaxes(range=[0, 1], row=4, col=1)

            fig.add_shape(type="line", x0=-0.0, y0=0.920, x1=1.0, y1=0.920, line=dict(color="LightSeaGreen", width=4, dash="dot",))

            fig.add_shape(type="line", x0=-0.0, y0=0.640, x1=1.0, y1=0.640,
                          line=dict(color="LightSeaGreen", width=4, dash="dot", ))

            fig.add_shape(type="line", x0=-0.0, y0=0.357, x1=1.0, y1=0.357,
                          line=dict(color="LightSeaGreen", width=4, dash="dot", ))

            fig.add_shape(type="line", x0=-0.0, y0=0.078, x1=1.0, y1=0.078,
                          line=dict(color="LightSeaGreen", width=4, dash="dot", ))

            fig.update_xaxes(showticklabels=False)  # hide all the xticks
            fig.update_xaxes(showticklabels=True, row=4, col=1)

            # fig.update_layout(shapes=[
            #     dict(
            #         type='line',
            #         color="MediumPurple",
            #         yref='paper', y0=0.945, y1=0.945,
            #         xref='x', x0=-0.5, x1=7.5
            #     )
            # ])
            fig.update_yaxes(showgrid=True, gridwidth=1)
            fig.update_xaxes(showgrid=True, gridwidth=1)

            filename = "%s/ML_performance.html" % (out_dir)
            print(filename)
            fig.write_html(filename)
            # fig.show()

