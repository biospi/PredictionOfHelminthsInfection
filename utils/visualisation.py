import glob
import os

import matplotlib
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cwt._cwt import CWT, plotLine, STFT, plot_cwt_power, plot_stft_power
from utils.Utils import create_rec_dir, anscombe
import random
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme
from tqdm import tqdm
from pathlib import Path
from BaselineRemoval import BaselineRemoval

from utils._normalisation import CenterScaler


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def add_separator(df_):
    df_ = df_.reset_index(drop=True)
    idxs = []
    d = df_["animal_ids"].values
    for i in range(d.size - 1):
        if d[i] != d[i + 1]:
            idxs.append(i)
    df_ = df_.reindex(df_.index.values.tolist() + [str(x).zfill(5) + "a" for x in idxs])
    df_.index = [str(x).zfill(5) for x in df_.index]
    df_ = df_.sort_index()
    ni = pd.Series(df_["animal_ids"].astype(np.float).values).interpolate(method='nearest').values
    df_["animal_ids"] = ni.tolist()
    nt = pd.Series(df_["target"].astype(np.float).values).interpolate(method='nearest').values
    df_["target"] = nt.astype(int).tolist()
    return df_


def plot_groups(N_META, animal_ids, class_healthy_label, class_unhealthy_label, class_healthy, class_unhealthy, graph_outputdir,
                df, title="title", xlabel='xlabel', ylabel='target',
                ntraces=1, idx_healthy=None, idx_unhealthy=None,
                show_max=True, show_min=False, show_mean=True, show_median=True, stepid=0):
    """Plot all rows in dataframe for each class Health or Unhealthy.

    Keyword arguments:
    df -- input dataframe containing samples (activity data, label/target)
    """
    df_healthy = df[df["target"] == class_healthy].iloc[:, :-N_META].values
    df_unhealthy = df[df["target"] == class_unhealthy].iloc[:, :-N_META].values

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(34.80, 7.20))
    fig.suptitle(title, fontsize=18)

    ymin = np.min(df.iloc[:, :-N_META].values)
    if idx_healthy is None or idx_unhealthy is None:
        ymax = np.max(df.iloc[:, :-N_META].values)
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
            ax1.set_title("Healthy(%s) animals %d / displaying %d" % (
                class_healthy_label, df_healthy.shape[0], df_healthy.shape[0]))
        else:
            ax1.set_title(
                "Healthy(%s) animals %d / displaying %d" % (class_healthy_label, df_healthy.shape[0], ntraces))
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
            ax2.set_title("Unhealthy(%s) %d samples / displaying %d" % (
                class_unhealthy_label, df_unhealthy.shape[0], df_unhealthy.shape[0]))
        else:
            ax2.set_title(
                "Unhealthy(%s) animals %d / displaying %d" % (class_unhealthy_label, df_unhealthy.shape[0], ntraces))
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

    # plt.show()
    filename = "%d_%s.png" % (stepid, title.replace(" ", "_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    # print('saving fig...')
    fig.savefig(filepath)
    fig.savefig(filepath.replace("png", "svg"))
    # print("saved!")

    print("building heatmaps...")
    cbarlocs = [.81, .19]
    # add row separator
    df_ = df.copy()
    df_["animal_ids"] = animal_ids

    df_healthy_ = add_separator(df_[df_["target"] == class_healthy])
    df_unhealthy_ = add_separator(df_[df_["target"] == class_unhealthy])

    t1 = "Healthy(%s) %d animals  %d samples" % (
        class_healthy_label, df_healthy_["animal_ids"].astype(str).drop_duplicates().size, df_healthy_.shape[0])
    t2 = "UnHealthy(%s) %d animals %d samples" % (
        class_unhealthy_label, df_unhealthy_["animal_ids"].astype(str).drop_duplicates().size, df_unhealthy_.shape[0])
    fig_ = make_subplots(rows=2, cols=1, x_title=xlabel, y_title="Transponder", subplot_titles=(t1, t2))
    fig_.add_trace(
        go.Heatmap(
            z=df_healthy_.iloc[:, :-2],
            x=ticks,
            y=[str(int(float(x[0]))) + "_" + str(x[1]) for x in
               zip(df_healthy_["animal_ids"].astype(str).tolist(), list(range(df_healthy_.shape[0])))],
            colorbar=dict(len=0.40, y=cbarlocs[0]),
            colorscale='Viridis'),
        row=1, col=1
    )

    fig_.add_trace(
        go.Heatmap(
            z=df_unhealthy_.iloc[:, :-2],
            x=ticks,
            y=[str(int(float(x[0]))) + "_" + str(x[1]) for x in
               zip(df_unhealthy_["animal_ids"].astype(str).tolist(), list(range(df_unhealthy_.shape[0])))],
            colorbar=dict(len=0.40, y=cbarlocs[1]),
            colorscale='Viridis'),
        row=2, col=1
    )
    fig_['layout']['xaxis']['tickformat'] = '%H:%M'
    fig_['layout']['xaxis2']['tickformat'] = '%H:%M'

    zmin = min([np.min(df_unhealthy.flatten()), np.min(df_unhealthy.flatten())])
    zmax = max([np.max(df_unhealthy.flatten()), np.max(df_unhealthy.flatten())])

    fig_.data[0].update(zmin=zmin, zmax=zmax)
    fig_.data[1].update(zmin=zmin, zmax=zmax)

    fig_.update_layout(title_text=title)
    filename = "%d_%s_heatmap.html" % (stepid, title.replace(" ", "_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    print(filepath)
    fig_.write_html(filepath)

    fig.clear()
    plt.close(fig)

    return idx_healthy, idx_unhealthy


def plot_2d_space(X, y, filename_2d_scatter, label_series, title='title'):
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    print("plot_2d_space")
    if len(X[0]) == 1:
        for l in zip(np.unique(y)):
            ax.scatter(
                X[y == l, 0],
                np.zeros(X[y == l, 0].size),
                label=l
            )
    else:
        for l in zip(np.unique(y)):
            ax.scatter(
                X[y == l[0]][:, 0],
                X[y == l[0]][:, 1],
                label=label_series[l[0]]
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


def plot_time_pca(meta_size, df, output_dir, label_series, title="title", y_col="label"):
    X = pd.DataFrame(PCA(n_components=2).fit_transform(df.iloc[:, :-meta_size])).values
    y = df["target"].astype(int)
    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = "%s/%s.png" % (output_dir, filename)
    plot_2d_space(X, y, filepath, label_series, title=title)


def plot_time_lda(N_META, df, output_dir, label_series, title="title", y_col="label"):
    y = df["target"].astype(int).values
    X = df.iloc[:, :-N_META].values
    n_components = np.unique(y).size - 1
    X = pd.DataFrame(LDA(n_components=n_components).fit_transform(X, y)).values
    # y = df_time_domain.iloc[:, -1].astype(int)
    filename = title.replace(" ", "_")
    filepath = "%s/%s.png" % (output_dir, filename)
    plot_2d_space(X, y, filepath, label_series, title=title)


def format(text):
    return text.replace("activity_no_norm", "TimeDom->")\
        .replace("activity_quotient_norm", "TimeDom->QN->")\
        .replace("cwt_quotient_norm", "TimeDom->QN->CWT->")\
        .replace("cwt_no_norm", "TimeDom->CWT->") \
        .replace("_", "->") \
        .replace("cwt_quotient_no_norm", "TimeDom->CWT->") \
        .replace("humidity", "Humidity->") \
        .replace("_humidity", "Humidity->") \
        .replace(",", "").replace("(", "").replace(")", "").replace("'","").replace(" ","").replace("->->","->").replace("_","->")


def stringArrayToArray(string):
    return [float(x) for x in string.replace("\n", "").replace("[", "").replace("]", "").replace(",", "").split(" ") if
     len(x) > 0]


def formatForBoxPlot(df):
    print("formatForBoxPlot...")
    dfs = []
    for index, row in df.iterrows():
        data = pd.DataFrame()
        test_balanced_accuracy_score = stringArrayToArray(row["test_balanced_accuracy_score"])
        test_precision_score0 = stringArrayToArray(row["test_precision_score0"])
        test_precision_score1 = stringArrayToArray(row["test_precision_score1"])
        test_recall_score0 = stringArrayToArray(row["test_recall_score0"])
        test_recall_score1 = stringArrayToArray(row["test_recall_score1"])
        test_f1_score0 = stringArrayToArray(row["test_f1_score0"])
        test_f1_score1 = stringArrayToArray(row["test_f1_score1"])
        roc_auc_scores = stringArrayToArray(row["roc_auc_scores"])
        config = [row["config"] for _ in range(len(test_balanced_accuracy_score))]
        data["test_balanced_accuracy_score"] = test_balanced_accuracy_score
        data["test_precision_score0"] = test_precision_score0
        data["test_precision_score1"] = test_precision_score1
        data["test_recall_score0"] = test_recall_score0
        data["test_recall_score1"] = test_recall_score1
        data["test_f1_score0"] = test_f1_score0
        data["test_f1_score1"] = test_f1_score1
        roc_auc_scores.extend([0] * (len(test_balanced_accuracy_score) - len(roc_auc_scores))) #in case auc could not be computed for fold
        data["roc_auc_scores"] = roc_auc_scores
        data["config"] = config
        dfs.append(data)
    formated = pd.concat(dfs, axis=0)
    return formated


def plotMlReportFinal(paths, output_dir):
    print("building report visualisation...")
    dfs = []
    label_dict = {}
    for path in paths:
        path = path.replace("\\", "/")
        target_label = path.split("/")[4].split("_")[-2]
        print(target_label)
        df = pd.read_csv(str(path), index_col=None)
        medians = []
        for value in df["roc_auc_scores"].values:
            v = stringArrayToArray(value)
            medians.append(np.median(v))
        df["median_auc"] = medians

        df["config"] = ["%s->" % target_label.upper() + "%dDAYS->" % df["days"].values[0] + format(str(x)) for x in list(zip(df.steps, df.classifier))]
        df = df.sort_values('median_auc')
        df = df.drop_duplicates(subset=['config'], keep='first')
        label_dict[target_label] = df["class1"].values[0]
        label_dict[df["class_0_label"].values[0]] = df["class0"].values[0]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.sort_values('median_auc')

    t4 = "AUC performance of different inputs<br>%s" % str(label_dict)

    t3 = "Accuracy performance of different inputs<br>%s" % str(label_dict)

    t1 = "Precision class0 performance of different inputs<br>%s" % str(label_dict)

    t2 = "Precision class1 performance of different inputs<br>%s" % str(label_dict)

    fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))

    df = formatForBoxPlot(df)

    fig.append_trace(px.box(df, x='config', y='test_precision_score0').data[0], row=1, col=1)
    fig.append_trace(px.box(df, x='config', y='test_precision_score1').data[0], row=2, col=1)
    fig.append_trace(px.box(df, x='config', y='test_balanced_accuracy_score').data[0], row=3, col=1)
    fig.append_trace(px.box(df, x='config', y='roc_auc_scores').data[0], row=4, col=1)

    # fig.update_yaxes(range=[df["precision_score0_mean"].min()/1.1, 1], row=1, col=1)
    # fig.update_yaxes(range=[df["precision_score1_mean"].min()/1.1, 1], row=2, col=1)
    # fig.update_yaxes(range=[df["balanced_accuracy_score_mean"].min()/1.1, 1], row=3, col=1)
    # fig.update_yaxes(range=[df["roc_auc_score_mean"].min()/1.1, 1], row=4, col=1)

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
    filepath = output_dir + "/" + "ML_performance_final.html"
    print(filepath)
    fig.write_html(filepath)
    # fig.show()


def plotMlReport(path, output_dir):
    print("building report visualisation...")
    df = pd.read_csv(str(path), index_col=None)
    medians = []
    for value in df["roc_auc_scores"].values:
        v = stringArrayToArray(value)
        medians.append(np.median(v))
    df["median_auc"] = medians

    df["config"] = [format(str(x)) for x in list(zip(df.steps, df.classifier))]
    df = df.sort_values('median_auc')
    df = df.drop_duplicates(subset=['config'], keep='first')
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

    df = formatForBoxPlot(df)

    fig.append_trace(px.box(df, x='config', y='test_precision_score0').data[0], row=1, col=1)
    fig.append_trace(px.box(df, x='config', y='test_precision_score1').data[0], row=2, col=1)
    fig.append_trace(px.box(df, x='config', y='test_balanced_accuracy_score').data[0], row=3, col=1)
    fig.append_trace(px.box(df, x='config', y='roc_auc_scores').data[0], row=4, col=1)

    # fig.update_yaxes(range=[df["precision_score0_mean"].min()/1.1, 1], row=1, col=1)
    # fig.update_yaxes(range=[df["precision_score1_mean"].min()/1.1, 1], row=2, col=1)
    # fig.update_yaxes(range=[df["balanced_accuracy_score_mean"].min()/1.1, 1], row=3, col=1)
    # fig.update_yaxes(range=[df["roc_auc_score_mean"].min()/1.1, 1], row=4, col=1)

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
    filepath = output_dir + "/" + "ML_performance.html"
    print(filepath)
    fig.write_html(filepath)
    # fig.show()


def plot_zeros_distrib(label_series, data_frame_no_norm, graph_outputdir, title="Percentage of zeros in activity per sample"):
    print("plot_zeros_distrib...")
    data = {}
    target_labels = []
    z_prct = []

    for index, row in data_frame_no_norm.iterrows():
        a = row[:-1].values
        label = label_series[row[-1]]

        target_labels.append(label)
        z_prct.append(np.sum(a == np.log(anscombe(0))) / len(a))

        if label not in data.keys():
            data[label] = a
        else:
            data[label] = np.append(data[label], a)
    distrib = {}
    for key, value in data.items():
        zeros_count = np.sum(value == np.log(anscombe(0))) / len(value)
        lcount = np.sum(data_frame_no_norm["target"] == {v: k for k, v in label_series.items()}[key])
        distrib[str(key) + " (%d)" % lcount] = zeros_count

    plt.bar(range(len(distrib)), list(distrib.values()), align='center')
    plt.xticks(range(len(distrib)), list(distrib.keys()))
    plt.title(title)
    plt.xlabel('Famacha samples (number of sample in class)')
    plt.ylabel('Percentage of zero values in samples')
    # plt.show()
    print(distrib)

    df = pd.DataFrame.from_dict({'Percent of zeros': z_prct, 'Target': target_labels})
    df.to_csv(graph_outputdir + "/z_prct_data.data")
    g = (ggplot(df)  # defining what data to use
         + aes(x='Target', y='Percent of zeros', color='Target', shape='Target')  # defining what variable to use
         + geom_jitter()  # defining the type of plot to use
         + stat_summary(geom="crossbar", color="black", width=0.2)
         + theme(subplots_adjust={'right': 0.82})
         )

    fig = g.draw()
    fig.tight_layout()
    # fig.show()
    filename = "zero_percent_%s.png" % title.lower().replace(" ","_")
    filepath = "%s/%s" % (graph_outputdir, filename)
    # print('saving fig...')
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


def plotHeatmap(X, out_dir="", title="Heatmap", filename="heatmap.html", y_log=False, yaxis="", xaxis="Time in minutes"):
    # fig = make_subplots(rows=len(transponders), cols=1)
    ticks = list(range(X.shape[1]))
    fig = make_subplots(rows=1, cols=1)
    if y_log:
        X_log = np.log(anscombe(X))
    trace = go.Heatmap(
            z=X_log if y_log else X,
            x=ticks,
            y=list(range(X.shape[0])),
            colorscale='Viridis')
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)
    #fig.show()
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_pr_range(ax_pr, y_ground_truth, y_proba, aucs, out_dir, classifier_name, fig, cv_name, days):
    y_ground_truth = np.concatenate(y_ground_truth)
    y_proba = np.concatenate(y_proba)
    mean_precision, mean_recall, _ = precision_recall_curve(y_ground_truth, y_proba)

    mean_auc = auc(mean_recall, mean_precision)
    lo, hi = mean_confidence_interval(aucs)
    label = r'Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )' % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r'Mean ROC (Mean AUC = %0.2f)' % mean_auc
    ax_pr.step(mean_recall, mean_precision, label=label, lw=2, color='black')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='lower left', fontsize='small')

    ax_pr.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
              title="Precision Recall curve days=%d cv=%s" % (days, cv_name))
    ax_pr.legend(loc="lower right")
    # fig.show()
    path = "%s/pr_curve/%s/" % (out_dir, cv_name)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'pr_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)

    # path = "%s/roc_curve/svg/" % out_dir
    # create_rec_dir(path)
    # final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    # print(final_path)
    # fig.savefig(final_path)
    return mean_auc


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig, cv_name, days):
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
    ax.plot(mean_fpr, mean_tpr, color='black',
            label=label,
            lw=2, alpha=1)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic days=%d cv=%s" % (days, cv_name))
    ax.legend(loc="lower right")
    # fig.show()
    path = "%s/roc_curve/%s/" % (out_dir, cv_name)
    create_rec_dir(path)
    final_path = '%s/%s' % (path, 'roc_%s.png' % classifier_name)
    print(final_path)
    fig.savefig(final_path)

    # path = "%s/roc_curve/svg/" % out_dir
    # create_rec_dir(path)
    # final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    # print(final_path)
    # fig.savefig(final_path)
    return mean_auc


def plotDistribution(X, output_dir, filename):
    print("plot data distribution...")
    hist_array = X.flatten()
    hist_array_nrm = hist_array[~np.isnan(hist_array)]
    df = pd.DataFrame(hist_array_nrm, columns=["value"])
    fig = px.histogram(df, x="value", nbins=np.unique(hist_array_nrm).size, title=filename)
    filename = output_dir + "/" + "%s.html" % filename
    create_rec_dir(filename)
    fig.write_html(filename)


def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")

def rolling_window(array, window_size,freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]


def plotMeanGroups(n_scales, sfft_window, wavelet_f0, df, label_series, N_META, out_dir, filename="mean_of_groups.html"):
    print("plot mean group...")
    traces = []
    fig_group_means = go.Figure()
    fig_group_median = go.Figure()
    for key in tqdm(label_series.keys()):
        df_ = df[df["target"] == key]
        fig_group = go.Figure()
        n = df_.shape[0]
        for index, row in df_.iterrows():
            x = np.arange(row.shape[0]-N_META)
            y = row.iloc[:-N_META].values
            id = str(int(float(row.iloc[-4])))
            date = row.iloc[-2]
            label = label_series[key]
            name = "%s %s %s" % (id, date, label)
            fig_group.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))
        mean = np.mean(df_.iloc[:, :-N_META], axis=0)
        median = np.median(df_.iloc[:, :-N_META], axis=0)

        s = mean.values
        s = anscombe(s)
        s = np.log(s)
        #s = StandardScaler().fit_transform(s.reshape(-1, 1)).flatten()
        #s = MinMaxScaler(feature_range=(0, 1)).fit_transform(s.reshape(-1, 1)).flatten()
        #s = BaselineRemoval(s).ZhangFit()
        #s = sklearn.preprocessing.normalize(s)
        #s = BaselineRemoval(s).ModPoly(2)

        # stop = s.copy()
        # stop[stop >= 0] = 0
        # stop = CenterScaler().transform(stop)
        #
        # sbottom = s.copy()
        # sbottom[sbottom < 0] = 0
        # sbottom = CenterScaler().transform(sbottom)

        plotLine(np.array([s]), out_dir+"/", label + "_" + str(df_.shape[0]), label + "_" + str(df_.shape[0])+".html")
        #
        # slices = rolling_window(s, 400, 400)
        # for i, s in enumerate(slices):
        s = CenterScaler(divide_by_std=False).transform(s)
        i = 0
        if wavelet_f0 is not None:
            CWT(hd=True, wavelet_f0=wavelet_f0, out_dir=out_dir+"/", step_slug=label + "_" + str(df_.shape[0]) +"_"+str(i),
                animal_ids=[], targets=[], dates=[], n_scales=n_scales).transform([s])

        if sfft_window is not None:
            STFT(sfft_window=sfft_window, out_dir=out_dir+"/", step_slug="ANSCOMBE_" + label + "_" + str(df_.shape[0]),
                animal_ids=[], targets=[], dates=[]).transform([s])

        fig_group.add_trace(go.Scatter(x=x, y=mean, mode='lines', name="Mean (%d) %s" % (n, label), line_color='#000000'))
        fig_group_means.add_trace(go.Scatter(x=x, y=mean, mode='lines', name="Mean (%d) %s" % (n, label)))
        fig_group_median.add_trace(go.Scatter(x=x, y=median, mode='lines', name="Median (%d) %s" % (n, label)))
        fig_group.update_layout(
            title="%d samples in category %s" % (n, label),
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)"
        )
        fig_group_means.update_layout(
            title="Mean of samples for each category",
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)"
        )
        fig_group_median.update_layout(
            title="Median of samples for each category",
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)"
        )
        traces.append(fig_group)
        #fig_group.show()

    traces.append(fig_group_means)
    traces.append(fig_group_median)
    traces = traces[::-1] #put the median grapth first

    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    figures_to_html(traces, filename=file_path)


def plot_mosaic(cv_name, directory_t, filename, subdir):

    cv_dir = []
    for item in directory_t:
        roc_dir_path = "%s/roc_curve" % item
        if cv_name in item:
            cv_dir.append(roc_dir_path)

    images = []
    for i, item in enumerate(cv_dir):
        files_roc = [str(x) for x in Path(item).rglob('*.png')]
        files_pr = [str(x) for x in Path(item.replace("roc_curve", "pr_curve")).rglob('*.png')]
        for j in range(len(files_roc)):
            images.append(files_roc[j])
            images.append(files_pr[j])

    steps = []
    cwt_meta = []
    stft_meta = []
    for image in images:
        steps.append("_".join(image.split("\\")[-1].replace(".png", "").split("_")[1:]))
        cwt_meta.append("wf0"+image.split("\\")[-4].split("wf0")[-1] if "wf0" in image else "")
        stft_meta.append("window" + image.split("\\")[-4].split("window")[-1] if "window" in image else "")

    df = pd.DataFrame()
    df["file"] = images
    df["step"] = steps
    df["wf0"] = cwt_meta
    df["window"] = stft_meta
    list_of_df = [g for _, g in df.groupby(['step'])]

    for dfs_g in list_of_df:
        for t in ['wf0', 'window']:
            for group in [g for _, g in dfs_g.groupby(t)]:
                if group.shape[0] != 14:
                    continue
                images = group["file"].values
                step_name = group["step"].values[0]
                if step_name == "":
                    step_name = "None"
                cv_name = group["file"].values[0].split("\\")[-2]
                wavelet_meta = group["wf0"].values[0]
                stft_meta = group["window"].values[0]

                fig = plt.figure(figsize=(30.0, 35.0))
                fig.suptitle('SVM performances for activity dataset (1day ... 7days)\nCross validation=%s | Preprocessing steps=%s' % (cv_name, step_name), fontsize=30)

                columns = 2
                rows = int(np.ceil(len(images)/2))
                for i, path in enumerate(images):
                    img = plt.imread(path)
                    fig.add_subplot(rows, columns, i + 1)
                    plt.imshow(img)
                    plt.title = path
                    plt.axis('off')
                fig.tight_layout()
                filepath = "%s/roc_pr_curves_%s/%s" % (output_dir, subdir, step_name + "_" + wavelet_meta + "_" + stft_meta + "_" + filename)
                create_rec_dir(filepath)
                print(filepath)
                fig.savefig(filepath)


def build_roc_mosaic(input_dir, output_dir):
    print("input_dir=", input_dir)
    dir_list = ["%s/%s" % (input_dir, name) for name in os.listdir(input_dir) if "ml_" in name]
    dir_list_1to2 = []
    dir_list_2to2 = []
    for path in dir_list:
        if "1to2" in path.lower():
            dir_list_1to2.append(path)
        if "2to2" in path.lower():
            dir_list_2to2.append(path)

    # plot_mosaic("kfold", dir_list_1to2, "1to2_roc_pr_curves_kfold.png", "1to2")
    # plot_mosaic("l2out", dir_list_1to2, "1to2_roc_pr_curves_l2outd.png", "1to2")
    # plot_mosaic("l1out",dir_list_1to2,  "1to2_roc_pr_curves_l1out.png", "1to2")

    plot_mosaic("kfold", dir_list_2to2, "2to2_roc_pr_curves_kfold.png", "2to2")
    plot_mosaic("l2out", dir_list_2to2, "2to2_roc_pr_curves_l2outd.png", "2to2")
    plot_mosaic("l1out", dir_list_2to2,  "2to2_roc_pr_curves_l1out.png", "2to2")


def SampleVisualisation(df, shape, N_META, out_dir, step_slug, sfft_window, stft_time, scales):
    print("sample visualisation...")
    for i, row in df.iterrows():
        activity = row[:-N_META]
        target = row["target"]

        date = datetime.strptime(row["date"], '%d/%m/%Y').strftime('%d_%m_%Y')
        epoch = str(int(datetime.strptime(row["date"], '%d/%m/%Y').timestamp()))

        imputed_days = row["imputed_days"]
        animal_id = row["id"]

        if "CWT" in step_slug:
            cwt = activity.values.reshape(shape).astype(np.float)
            plot_cwt_power(None, None, epoch, date, animal_id, target, step_slug, out_dir, i, activity, cwt.copy(), None, scales, log_yaxis=False, standard_scale=True, format_xaxis=False)

        if "STFT" in step_slug:
            power_sfft = activity.values.reshape(shape).astype(np.float)
            plot_stft_power(sfft_window, stft_time, epoch, date, animal_id, target, step_slug, out_dir, i, activity,
                            power_sfft, scales, format_xaxis=False
                            , vmin=None, vmax=None, standard_scale=True)


if __name__ == "__main__":
    dir_path = "F:/Data2/job_debug/ml"
    output_dir = "F:/Data2/job_debug/ml"
    build_roc_mosaic(dir_path, output_dir)