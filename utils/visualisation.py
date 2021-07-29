import glob
import os
import pathlib

import matplotlib
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cwt._cwt import CWT, plotLine, STFT, plot_cwt_power, plot_stft_power
from utils.Utils import anscombe
import random
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme
from tqdm import tqdm
from pathlib import Path

from utils._normalisation import CenterScaler

import numpy as np
import matplotlib.pyplot as plt


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
    ni = (
        pd.Series(df_["animal_ids"].astype(np.float).values)
        .interpolate(method="nearest")
        .values
    )
    df_["animal_ids"] = ni.tolist()
    nt = (
        pd.Series(df_["target"].astype(np.float).values)
        .interpolate(method="nearest")
        .values
    )
    df_["target"] = nt.astype(int).tolist()
    return df_


def plot_groups(
    N_META,
    animal_ids,
    class_healthy_label,
    class_unhealthy_label,
    class_healthy,
    class_unhealthy,
    graph_outputdir,
    df,
    title="title",
    xlabel="xlabel",
    ylabel="target",
    ntraces=1,
    idx_healthy=None,
    idx_unhealthy=None,
    show_max=True,
    show_min=False,
    show_mean=True,
    show_median=True,
    stepid=0,
):
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
        ymax = max(
            [np.max(df_healthy[idx_healthy]), np.max(df_unhealthy[idx_unhealthy])]
        )

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
            ax1.set_title(
                "Healthy(%s) animals %d / displaying %d"
                % (class_healthy_label, df_healthy.shape[0], df_healthy.shape[0])
            )
        else:
            ax1.set_title(
                "Healthy(%s) animals %d / displaying %d"
                % (class_healthy_label, df_healthy.shape[0], ntraces)
            )
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
    if idx_unhealthy is None:
        idx_unhealthy = random.sample(range(1, df_unhealthy.shape[0]), ntraces)
    for i in idx_unhealthy:
        ax2.plot(ticks, df_unhealthy[i])
        ax2.set(xlabel=xlabel, ylabel=ylabel)
        ax2.set_xticklabels(ticks, fontsize=12)
        if ntraces is None:
            ax2.set_title(
                "Unhealthy(%s) %d samples / displaying %d"
                % (class_unhealthy_label, df_unhealthy.shape[0], df_unhealthy.shape[0])
            )
        else:
            ax2.set_title(
                "Unhealthy(%s) animals %d / displaying %d"
                % (class_unhealthy_label, df_unhealthy.shape[0], ntraces)
            )
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
    if show_max:
        # ax1.plot(ticks, np.amax(df_healthy, axis=0), c='tab:gray', label='max', linestyle='-')
        # ax2.plot(ticks, np.amax(df_unhealthy, axis=0), c='tab:gray', label='max', linestyle='-')
        ax1.fill_between(
            ticks,
            np.amax(df_healthy, axis=0),
            color="lightgrey",
            label="max",
            zorder=-1,
        )
        ax2.fill_between(
            ticks, np.amax(df_unhealthy, axis=0), label="max", color="lightgrey"
        )
        ax1.legend()
        ax2.legend()
    if show_min:
        ax1.plot(ticks, np.amin(df_healthy, axis=0), c="red", label="min")
        ax2.plot(ticks, np.amin(df_unhealthy, axis=0), c="red", label="min")
        ax1.legend()
        ax2.legend()

    if show_mean:
        ax1.plot(
            ticks,
            np.mean(df_healthy, axis=0),
            c="black",
            label="mean",
            alpha=1,
            linestyle="-",
        )
        ax2.plot(
            ticks,
            np.mean(df_unhealthy, axis=0),
            c="black",
            label="mean",
            alpha=1,
            linestyle="-",
        )
        ax1.legend()
        ax2.legend()

    if show_median:
        ax1.plot(
            ticks,
            np.median(df_healthy, axis=0),
            c="black",
            label="median",
            alpha=1,
            linestyle=":",
        )
        ax2.plot(
            ticks,
            np.median(df_unhealthy, axis=0),
            c="black",
            label="median",
            alpha=1,
            linestyle=":",
        )
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
    cbarlocs = [0.81, 0.19]
    # add row separator
    df_ = df.copy()
    df_["animal_ids"] = animal_ids

    df_healthy_ = add_separator(df_[df_["target"] == class_healthy])
    df_unhealthy_ = add_separator(df_[df_["target"] == class_unhealthy])

    t1 = "Healthy(%s) %d animals  %d samples" % (
        class_healthy_label,
        df_healthy_["animal_ids"].astype(str).drop_duplicates().size,
        df_healthy_.shape[0],
    )
    t2 = "UnHealthy(%s) %d animals %d samples" % (
        class_unhealthy_label,
        df_unhealthy_["animal_ids"].astype(str).drop_duplicates().size,
        df_unhealthy_.shape[0],
    )
    fig_ = make_subplots(
        rows=2, cols=1, x_title=xlabel, y_title="Transponder", subplot_titles=(t1, t2)
    )
    fig_.add_trace(
        go.Heatmap(
            z=df_healthy_.iloc[:, :-2],
            x=ticks,
            y=[
                str(int(float(x[0]))) + "_" + str(x[1])
                for x in zip(
                    df_healthy_["animal_ids"].astype(str).tolist(),
                    list(range(df_healthy_.shape[0])),
                )
            ],
            colorbar=dict(len=0.40, y=cbarlocs[0]),
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )

    fig_.add_trace(
        go.Heatmap(
            z=df_unhealthy_.iloc[:, :-2],
            x=ticks,
            y=[
                str(int(float(x[0]))) + "_" + str(x[1])
                for x in zip(
                    df_unhealthy_["animal_ids"].astype(str).tolist(),
                    list(range(df_unhealthy_.shape[0])),
                )
            ],
            colorbar=dict(len=0.40, y=cbarlocs[1]),
            colorscale="Viridis",
        ),
        row=2,
        col=1,
    )
    fig_["layout"]["xaxis"]["tickformat"] = "%H:%M"
    fig_["layout"]["xaxis2"]["tickformat"] = "%H:%M"

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


def plot_2d_space(X, y, filename_2d_scatter, label_series, title="title"):
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    print("plot_2d_space")
    if len(X[0]) == 1:
        for l in zip(np.unique(y)):
            ax.scatter(X[y == l, 0], np.zeros(X[y == l, 0].size), label=l)
    else:
        for l in zip(np.unique(y)):
            ax.scatter(X[y == l[0]][:, 0], X[y == l[0]][:, 1], label=label_series[l[0]])

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    print(filename_2d_scatter)
    folder = Path(filename_2d_scatter).parent
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename_2d_scatter)
    # plt.show()
    plt.close(fig)
    plt.clf()


def plot_time_pca(
    meta_size, df, output_dir, label_series, title="title", y_col="label"
):
    X = pd.DataFrame(PCA(n_components=2).fit_transform(df.iloc[:, :-meta_size])).values
    y = df["target"].astype(int)
    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = output_dir / filename
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
    return (
        text.replace("activity_no_norm", "TimeDom->")
        .replace("activity_quotient_norm", "TimeDom->QN->")
        .replace("cwt_quotient_norm", "TimeDom->QN->CWT->")
        .replace("cwt_no_norm", "TimeDom->CWT->")
        .replace("_", "->")
        .replace("cwt_quotient_no_norm", "TimeDom->CWT->")
        .replace("humidity", "Humidity->")
        .replace("_humidity", "Humidity->")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace(" ", "")
        .replace("->->", "->")
        .replace("_", "->")
    )


def stringArrayToArray(string):
    return [
        float(x)
        for x in string.replace("\n", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "")
        .split(" ")
        if len(x) > 0
    ]


def formatForBoxPlot(df):
    print("formatForBoxPlot...")
    dfs = []
    for index, row in df.iterrows():
        data = pd.DataFrame()
        test_balanced_accuracy_score = stringArrayToArray(
            row["test_balanced_accuracy_score"]
        )
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
        roc_auc_scores.extend(
            [0] * (len(test_balanced_accuracy_score) - len(roc_auc_scores))
        )  # in case auc could not be computed for fold
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
        target_label = [x for x in path.split("/")[4].split("_") if "to" in x.lower()][
            0
        ]
        meta = path.split("/")[4].split("_")[-1] + "->"
        if "night" not in meta:
            meta = "entireday->"
        print(target_label)
        df = pd.read_csv(str(path), index_col=None)
        medians = []
        for value in df["roc_auc_scores"].values:
            v = stringArrayToArray(value)
            medians.append(np.median(v))
        df["median_auc"] = medians

        df["config"] = [
            (
                "%s" % meta
                + "%s->" % target_label.upper()
                + "%dDAYS->" % df["days"].values[0]
                + format(str(x))
            )
            .replace("STANDARDSCALER", "STSC")
            .replace("ANSCOMBE", "ANS")
            for x in list(zip(df.steps, df.classifier))
        ]
        df = df.sort_values("median_auc")
        df = df.drop_duplicates(subset=["config"], keep="first")
        label_dict[
            target_label.replace("1To1", "Healthy")
            .replace("1To2", "Unhealthy")
            .replace("2to2", "Unhealthy")
        ] = df["class1"].values[0]
        label_dict[
            df["class_0_label"]
            .values[0]
            .replace("1To1", "Healthy")
            .replace("1To2", "Unhealthy")
            .replace("2to2", "Unhealthy")
        ] = df["class0"].values[0]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.sort_values("median_auc")

    t4 = "AUC performance of different inputs<br>%s" % str(label_dict)

    t3 = "Accuracy performance of different inputs<br>%s" % str(label_dict)

    t1 = "Precision class0 performance of different inputs<br>%s" % str(label_dict)

    t2 = "Precision class1 performance of different inputs<br>%s" % str(label_dict)

    fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))
    fig_auc_only = make_subplots(rows=1, cols=1)

    df = formatForBoxPlot(df)

    fig.append_trace(
        px.box(df, x="config", y="test_precision_score0").data[0], row=1, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_precision_score1").data[0], row=2, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_balanced_accuracy_score").data[0], row=3, col=1
    )
    fig.append_trace(px.box(df, x="config", y="roc_auc_scores").data[0], row=4, col=1)

    fig_auc_only.append_trace(
        px.box(df, x="config", y="roc_auc_scores", title=t4).data[0], row=1, col=1
    )

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
    fig.update_xaxes(showticklabels=True, row=4, col=1, automargin=True)

    # fig.update_layout(shapes=[
    #     dict(
    #         type='line',
    #         color="MediumPurple",
    #         yref='paper', y0=0.945, y1=0.945,
    #         xref='x', x0=-0.5, x1=7.5
    #     )
    # ])
    fig.update_yaxes(showgrid=True, gridwidth=1, automargin=True)
    fig.update_xaxes(showgrid=True, gridwidth=1, automargin=True)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=500))
    fig_auc_only.update_layout(margin=dict(l=20, r=20, t=20, b=500))
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "ML_performance_final.html"
    print(filepath)
    fig.write_html(str(filepath))
    filepath = output_dir / "ML_performance_final_auc.html"
    print(filepath)
    fig_auc_only.write_html(str(filepath))
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
    df = df.sort_values("median_auc")
    df = df.drop_duplicates(subset=["config"], keep="first")
    print(df)
    t4 = "AUC performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
        df["days"].values[0],
        df["class0"].values[0],
        df["class_0_label"].values[0],
        df["class1"].values[0],
        df["class_1_label"].values[0],
    )

    t3 = (
        "Accuracy performance of different inputs<br>Days=%d class0=%d %s class1=%d %s"
        % (
            df["days"].values[0],
            df["class0"].values[0],
            df["class_0_label"].values[0],
            df["class1"].values[0],
            df["class_1_label"].values[0],
        )
    )

    t1 = "Precision class0 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
        df["days"].values[0],
        df["class0"].values[0],
        df["class_0_label"].values[0],
        df["class1"].values[0],
        df["class_1_label"].values[0],
    )

    t2 = "Precision class1 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
        df["days"].values[0],
        df["class0"].values[0],
        df["class_0_label"].values[0],
        df["class1"].values[0],
        df["class_1_label"].values[0],
    )

    fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))

    df = formatForBoxPlot(df)

    fig.append_trace(
        px.box(df, x="config", y="test_precision_score0").data[0], row=1, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_precision_score1").data[0], row=2, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_balanced_accuracy_score").data[0], row=3, col=1
    )
    fig.append_trace(px.box(df, x="config", y="roc_auc_scores").data[0], row=4, col=1)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "ML_performance_2.html"
    print(filepath)
    fig.write_html(str(filepath))
    # fig.show()


def plot_zeros_distrib(
    label_series,
    data_frame_no_norm,
    graph_outputdir,
    title="Percentage of zeros in activity per sample",
):
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
        lcount = np.sum(
            data_frame_no_norm["target"] == {v: k for k, v in label_series.items()}[key]
        )
        distrib[str(key) + " (%d)" % lcount] = zeros_count

    plt.bar(range(len(distrib)), list(distrib.values()), align="center")
    plt.xticks(range(len(distrib)), list(distrib.keys()))
    plt.title(title)
    plt.xlabel("Famacha samples (number of sample in class)")
    plt.ylabel("Percentage of zero values in samples")
    # plt.show()
    print(distrib)

    df = pd.DataFrame.from_dict({"Percent of zeros": z_prct, "Target": target_labels})
    graph_outputdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(graph_outputdir / "z_prct_data.data")
    g = (
        ggplot(df)  # defining what data to use
        + aes(
            x="Target", y="Percent of zeros", color="Target", shape="Target"
        )  # defining what variable to use
        + geom_jitter()  # defining the type of plot to use
        + stat_summary(geom="crossbar", color="black", width=0.2)
        + theme(subplots_adjust={"right": 0.82})
    )

    fig = g.draw()
    fig.tight_layout()
    # fig.show()
    filename = f"zero_percent_{title.lower().replace(' ', '_')}.png"
    filepath = graph_outputdir / filename
    # print('saving fig...')
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


def plotAllFeatures(
    X,
    y,
    out_dir,
    title="Features visualisation",
    filename="heatmap.html",
    yaxis="value",
    xaxis="features",
):
    dfs = []
    for i in range(X.shape[0]):
        x = X[i, :]
        target = y[i]
        df = pd.DataFrame({"X": x, "y": target})
        dfs.append(df)
    df_data = pd.concat(dfs, axis=0)
    df_data = df_data.sort_index().reset_index()

    fig = px.line(df_data, x="index", y="X", color="y", line_dash="y")

    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)

    # fig.show()
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))


def plotHeatmap(
    X,
    out_dir,
    title="Heatmap",
    filename="heatmap.html",
    y_log=False,
    yaxis="",
    xaxis="Time in minutes",
):
    # fig = make_subplots(rows=len(transponders), cols=1)
    ticks = list(range(X.shape[1]))
    fig = make_subplots(rows=1, cols=1)
    if y_log:
        X_log = np.log(anscombe(X))
    trace = go.Heatmap(
        z=X_log if y_log else X,
        x=ticks,
        y=list(range(X.shape[0])),
        colorscale="Viridis",
    )
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)
    # fig.show()
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_pr_range(
    ax_pr, y_ground_truth, y_proba, aucs, out_dir, classifier_name, fig, cv_name, days
):
    y_ground_truth = np.concatenate(y_ground_truth)
    y_proba = np.concatenate(y_proba)
    mean_precision, mean_recall, _ = precision_recall_curve(y_ground_truth, y_proba)

    mean_auc = auc(mean_recall, mean_precision)
    lo, hi = mean_confidence_interval(aucs)
    label = r"Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )" % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r"Mean ROC (Mean AUC = %0.2f)" % mean_auc
    ax_pr.step(mean_recall, mean_precision, label=label, lw=2, color="black")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize="small")

    ax_pr.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Precision Recall curve days=%d cv=%s" % (days, cv_name),
    )
    ax_pr.legend(loc="lower right")
    # fig.show()

    path = out_dir / "pr_curve" / cv_name
    path.mkdir(parents=True, exist_ok=True)
    final_path = path / f"pr_{classifier_name}.png"
    print(final_path)
    fig.savefig(final_path)

    # path = "%s/roc_curve/svg/" % out_dir
    # create_rec_dir(path)
    # final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    # print(final_path)
    # fig.savefig(final_path)
    return mean_auc


def plot_roc_range(
    ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig, cv_name, days, info="None"
):
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)

    label = f"Mean ROC (Median AUC = {np.median(aucs):.2f}, 95%% CI [{lo:.4f}, {hi:.4f}] )"
    if len(aucs) <= 2:
        label = r"Mean ROC (Median AUC = %0.2f)" % np.median(aucs)
    ax.plot(mean_fpr, mean_tpr, color="black", label=label, lw=2, alpha=1)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"Receiver operating characteristic days:{days} cv:{cv_name} \n info:{info}"
    )
    ax.legend(loc="lower right")
    # fig.show()
    path = out_dir / "roc_curve" / cv_name
    path.mkdir(parents=True, exist_ok=True)
    final_path = path / f"roc_{classifier_name}.png"
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
    fig = px.histogram(
        df, x="value", nbins=np.unique(hist_array_nrm).size, title=filename
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{filename}.html"
    fig.write_html(str(filename))


def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, "w")
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def rolling_window(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]


def plotMeanGroups(
    n_scales,
    sfft_window,
    wavelet_f0,
    df,
    label_series,
    N_META,
    out_dir,
    filename="mean_of_groups.html",
):
    print("plot mean group...")
    traces = []
    fig_group_means = go.Figure()
    fig_group_median = go.Figure()
    for key in tqdm(label_series.keys()):
        df_ = df[df["target"] == key]
        fig_group = go.Figure()
        n = df_.shape[0]
        for index, row in df_.iterrows():
            x = np.arange(row.shape[0] - N_META)
            y = row.iloc[:-N_META].values
            id = str(int(float(row.iloc[-4])))
            date = row.iloc[-2]
            label = label_series[key]
            name = "%s %s %s" % (id, date, label)
            fig_group.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        mean = np.mean(df_.iloc[:, :-N_META], axis=0)
        median = np.median(df_.iloc[:, :-N_META], axis=0)

        s = mean.values
        s = anscombe(s)
        s = np.log(s)
        # s = StandardScaler().fit_transform(s.reshape(-1, 1)).flatten()
        # s = MinMaxScaler(feature_range=(0, 1)).fit_transform(s.reshape(-1, 1)).flatten()
        # s = BaselineRemoval(s).ZhangFit()
        # s = sklearn.preprocessing.normalize(s)
        # s = BaselineRemoval(s).ModPoly(2)

        # stop = s.copy()
        # stop[stop >= 0] = 0
        # stop = CenterScaler().transform(stop)
        #
        # sbottom = s.copy()
        # sbottom[sbottom < 0] = 0
        # sbottom = CenterScaler().transform(sbottom)

        plotLine(
            np.array([s]),
            out_dir,
            label + "_" + str(df_.shape[0]),
            label + "_" + str(df_.shape[0]) + ".html",
        )
        #
        # slices = rolling_window(s, 400, 400)
        # for i, s in enumerate(slices):
        s = CenterScaler(divide_by_std=False).transform(s)
        i = 0
        if wavelet_f0 is not None:
            CWT(
                hd=True,
                wavelet_f0=wavelet_f0,
                out_dir=out_dir,
                step_slug=label + "_" + str(df_.shape[0]) + "_" + str(i),
                animal_ids=[],
                targets=[],
                dates=[],
                n_scales=n_scales,
            ).transform([s])

        if sfft_window is not None:
            STFT(
                sfft_window=sfft_window,
                out_dir=out_dir,
                step_slug="ANSCOMBE_" + label + "_" + str(df_.shape[0]),
                animal_ids=[],
                targets=[],
                dates=[],
            ).transform([s])

        fig_group.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name="Mean (%d) %s" % (n, label),
                line_color="#000000",
            )
        )
        fig_group_means.add_trace(
            go.Scatter(x=x, y=mean, mode="lines", name="Mean (%d) %s" % (n, label))
        )
        fig_group_median.add_trace(
            go.Scatter(x=x, y=median, mode="lines", name="Median (%d) %s" % (n, label))
        )
        fig_group.update_layout(
            title="%d samples in category %s" % (n, label),
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)",
        )
        fig_group_means.update_layout(
            title="Mean of samples for each category",
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)",
        )
        fig_group_median.update_layout(
            title="Median of samples for each category",
            xaxis_title="Time in minute",
            yaxis_title="Activity (count)",
        )
        traces.append(fig_group)
        # fig_group.show()

    traces.append(fig_group_means)
    traces.append(fig_group_median)
    traces = traces[::-1]  # put the median grapth first
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    figures_to_html(traces, filename=str(file_path))


def plot_mosaic(cv_name, directory_t, filename, subdir):

    cv_dir = []
    for item in directory_t:
        roc_dir_path = "%s/roc_curve" % item
        if cv_name in item:
            cv_dir.append(roc_dir_path)

    images = []
    for i, item in enumerate(cv_dir):
        files_roc = [str(x) for x in Path(item).rglob("*.png")]
        files_pr = [
            str(x) for x in Path(item.replace("roc_curve", "pr_curve")).rglob("*.png")
        ]
        for j in range(len(files_roc)):
            images.append(files_roc[j])
            images.append(files_pr[j])

    steps = []
    cwt_meta = []
    stft_meta = []
    for image in images:
        steps.append("_".join(image.split("\\")[-1].replace(".png", "").split("_")[1:]))
        cwt_meta.append(
            "wf0" + image.split("\\")[-4].split("wf0")[-1] if "wf0" in image else ""
        )
        stft_meta.append(
            "window" + image.split("\\")[-4].split("window")[-1]
            if "window" in image
            else ""
        )

    df = pd.DataFrame()
    df["file"] = images
    df["step"] = steps
    df["wf0"] = cwt_meta
    df["window"] = stft_meta
    list_of_df = [g for _, g in df.groupby(["step"])]

    for dfs_g in list_of_df:
        for t in ["wf0", "window"]:
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
                fig.suptitle(
                    "SVM performances for activity dataset (1day ... 7days)\nCross validation=%s | Preprocessing steps=%s"
                    % (cv_name, step_name),
                    fontsize=30,
                )

                columns = 2
                rows = int(np.ceil(len(images) / 2))
                for i, path in enumerate(images):
                    img = plt.imread(path)
                    fig.add_subplot(rows, columns, i + 1)
                    plt.imshow(img)
                    plt.title = path
                    plt.axis("off")
                fig.tight_layout()
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = (
                    output_dir
                    / f"roc_pr_curves_{subdir}"
                    / f"{step_name}_{wavelet_meta}_{stft_meta}+{filename}"
                )

                print(filepath)
                fig.savefig(filepath)


def build_roc_mosaic(input_dir, output_dir):
    print("input_dir=", input_dir)
    dir_list = [
        "%s/%s" % (input_dir, name) for name in os.listdir(input_dir) if "ml_" in name
    ]
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
    plot_mosaic("l1out", dir_list_2to2, "2to2_roc_pr_curves_l1out.png", "2to2")


def SampleVisualisation(
    df, shape, N_META, out_dir, step_slug, sfft_window, stft_time, scales
):
    print("sample visualisation...")
    for i, row in df.iterrows():
        activity = row[:-N_META]
        target = row["target"]

        date = datetime.strptime(row["date"], "%d/%m/%Y").strftime("%d_%m_%Y")
        epoch = str(int(datetime.strptime(row["date"], "%d/%m/%Y").timestamp()))

        imputed_days = row["imputed_days"]
        animal_id = row["id"]

        if "CWT" in step_slug:
            cwt = activity.values.reshape(shape).astype(np.float)
            plot_cwt_power(
                None,
                None,
                epoch,
                date,
                animal_id,
                target,
                step_slug,
                out_dir,
                i,
                activity,
                cwt.copy(),
                None,
                scales,
                log_yaxis=False,
                standard_scale=True,
                format_xaxis=False,
            )

        if "STFT" in step_slug:
            power_sfft = activity.values.reshape(shape).astype(np.float)
            plot_stft_power(
                sfft_window,
                stft_time,
                epoch,
                date,
                animal_id,
                target,
                step_slug,
                out_dir,
                i,
                activity,
                power_sfft,
                scales,
                format_xaxis=False,
                vmin=None,
                vmax=None,
                standard_scale=True,
            )


def plot_3D_decision_boundaries(
    X,
    Y,
    train_x,
    train_y,
    test_x,
    test_y,
    title,
    clf,
    i,
    folder,
    sub_dir_name,
    auc,
    DR="PCA",
):
    Y = (Y != 1).astype(int)
    test_y = (test_y != 1).astype(int)
    train_y = (train_y != 1).astype(int)
    X = X[:, :3]  # we only take the first three features.

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = (
        lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y)
        / clf.coef_[0][2]
    )

    s = max([np.abs(X.max()), np.abs(X.min())])
    tmp = np.linspace(-s, s, 30)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure(figsize=(12.20, 7.20))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X[Y == 0, 0],
        X[Y == 0, 1],
        X[Y == 0, 2],
        marker="o",
        color="tab:blue",
        label="Class0 (Healthy)",
    )
    ax.scatter(
        X[Y == 1, 0],
        X[Y == 1, 1],
        X[Y == 1, 2],
        marker="s",
        color="tab:red",
        label="Class1 (Unhealthy)",
    )

    ax.scatter(
        test_x[test_y == 0, 0],
        test_x[test_y == 0, 1],
        test_x[test_y == 0, 2],
        marker="o",
        color="none",
        edgecolor="black",
        label="Test data Class0 (Healthy)",
    )
    ax.scatter(
        test_x[test_y == 1, 0],
        test_x[test_y == 1, 1],
        test_x[test_y == 1, 2],
        marker="s",
        color="none",
        edgecolor="black",
        label="Test data Class1 (Unhealthy)",
    )
    ax.set(
        xlabel="%s component 1" % DR,
        ylabel="%s component 2" % DR,
        zlabel="%s component 3" % DR,
    )
    handles, labels = ax.get_legend_handles_labels()
    # db_line = Line2D([0], [0], color=(183/255, 37/255, 42/255), label='Decision boundary')
    # handles.append(db_line)
    plt.legend(loc=4, fancybox=True, framealpha=0.4, handles=handles)
    plt.title(title + " AUC=%.2f" % auc)
    ttl = ax.title
    ttl.set_position([0.57, 0.97])

    ax.plot_surface(x, y, z(x, y), alpha=0.2)
    ax.view_init(30, 60)
    # plt.show()

    path = "%s/decision_boundaries_graphs/%s/" % (folder, sub_dir_name)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = "3D%sfold_%d.png" % (DR, i)
    final_path = "%s%s" % (path, filename)
    print(final_path)
    try:
        fig.savefig(final_path, bbox_inches="tight")
    except Exception as e:
        print(e)

    plt.close()
    # fig.show()
    plt.close()
    fig.clear()


# def plot_3D_decision_boundaries(train_x, train_y, test_x, test_y, title, clf, i, filename):
#     train_y = (train_y != 1).astype(int)
#     test_y = (test_y != 1).astype(int)
#
#     R('r3dDefaults$windowRect <- c(0,50, 1000, 1000) ')
#     R('open3d()')
#     plot3ddb = R('''
#     plot3ddb<-function(nnew, group, dat, kernel_, gamma_, coef_, cost_, tolerance_, probability_, test_x_, fitted_, title_, filepath){
#             set.seed(12345)
#             fit = svm(group ~ ., data=dat, kernel=kernel_, gamma=gamma_, coef0=coef_, cost=cost_, tolerance=tolerance_, fitted= fitted_, probability= probability_)
#             x = dat[,-1]$X1
#             y = dat[,-1]$X2
#             z = dat[,-1]$X3
#             x_test = test_x_[,-1]$X1
#             y_test = test_x_[,-1]$X2
#             z_test = test_x_[,-1]$X3
#             i <- 1
#             g = dat$group
#             x_1 <- list()
#             y_1 <- list()
#             z_1 <- list()
#             x_2 <- list()
#             y_2 <- list()
#             z_2 <- list()
#             for(var in g){
#                 if(!(x[i] %in% x_test) & !(y[i] %in% y_test)){
#                     if (var == 1){
#                         x_1 <- append(x_1, x[i])
#                         y_1 <- append(y_1, y[i])
#                         z_1 <- append(z_1, z[i])
#                     }else{
#                         x_2 <- append(x_2, x[i])
#                         y_2 <- append(y_2, y[i])
#                         z_2 <- append(z_2, z[i])
#                       }
#                 }
#               i <- i + 1
#             }
#
#             x_1 = as.numeric(x_1)
#             y_1 = as.numeric(y_1)
#             z_1 = as.numeric(z_1)
#
#             x_2 = as.numeric(x_2)
#             y_2 = as.numeric(y_2)
#             z_2 = as.numeric(z_2)
#
#
#             j <- 1
#             g_test = test_x_$class
#             x_1_test <- list()
#             y_1_test <- list()
#             z_1_test <- list()
#             x_2_test <- list()
#             y_2_test <- list()
#             z_2_test <- list()
#             for(var_test in g_test){
#               if (var_test == 1){
#                 x_1_test <- append(x_1_test, x_test[j])
#                 y_1_test <- append(y_1_test, y_test[j])
#                 z_1_test <- append(z_1_test, z_test[j])
#               }else{
#                 x_2_test <- append(x_2_test, x_test[j])
#                 y_2_test <- append(y_2_test, y_test[j])
#                 z_2_test <- append(z_2_test, z_test[j])
#               }
#
#               j <- j + 1
#             }
#
#             x_1_test = as.numeric(x_1_test)
#             y_1_test = as.numeric(y_1_test)
#             z_1_test = as.numeric(z_1_test)
#
#             x_2_test = as.numeric(x_2_test)
#             y_2_test = as.numeric(y_2_test)
#             z_2_test = as.numeric(z_2_test)
#
#             pch3d(x_2, y_2, z_2, pch = 24, bg = "#f19c51", color = "#f19c51", radius=0.4, alpha = 0.8)
#             pch3d(x_1, y_1, z_1, pch = 22, bg = "#6297bb", color = '#6297bb', radius=0.4, alpha = 1)
#
#             pch3d(x_1_test, y_1_test, z_1_test, pch = 22, bg = "#6297bb", color = 'red', radius=0.4, alpha = 0.8)
#             pch3d(x_2_test, y_2_test, z_2_test, pch = 24, bg = "#f19c51", color = "red", radius=0.4, alpha = 1)
#
#             newdat.list = lapply(test_x_[,-1], function(x) seq(min(x), max(x), len=nnew))
#             newdat      = expand.grid(newdat.list)
#             newdat.pred = predict(fit, newdata=newdat, decision.values=T)
#             newdat.dv   = attr(newdat.pred, 'decision.values')
#             newdat.dv   = array(newdat.dv, dim=rep(nnew, 3))
#             grid3d(c("x", "y+", "z"))
#             view3d(userMatrix = structure(c(0.850334823131561, -0.102673642337322,
#                                     0.516127586364746, 0, 0.526208400726318, 0.17674557864666,
#                                     -0.831783592700958, 0, -0.00582099659368396, 0.978886127471924,
#                                     0.20432074368, 0, 0, 0, 0, 1)))
#
#             decorate3d(box=F, axes = T, xlab = '', ylab='', zlab='', aspect = FALSE, expand = 1.03)
#             light3d(diffuse = "gray", specular = "gray")
#             contour3d(newdat.dv, level=0, x=newdat.list$X1, y=newdat.list$X2, z=newdat.list$X3, add=T, alpha=0.8, plot=T, smooth = 200, color='#28b99d', color2='#28b99d')
#             bgplot3d({
#                       plot.new()
#                       title(main = title_, line = -8, outer=F)
#                       #mtext(side = 1, 'This is a subtitle', line = 4)
#                       legend("bottomleft", inset=.1,
#                                pt.cex = 2,
#                                cex = 1,
#                                bty = "n",
#                                legend = c("Decision boundary", "Class 0", "Class 1", "Test data"),
#                                col = c("#28b99d", "#6297bb", "#f19c51", "red"),
#                                pch = c(15, 15,17, 1))
#             })
#             rgl.snapshot(filepath, fmt="png", top=TRUE)
#     }''')
#
#     nnew = test_x.shape[0]
#     gamma = clf._gamma
#     coef0 = clf.coef0
#     cost = clf.C
#     tolerance = clf.tol
#     probability_ = clf.probability
#
#     df = pd.DataFrame(train_x)
#     df.insert(loc=0, column='group', value=train_y + 1)
#     df.columns = ['group', 'X1', 'X2', 'X3']
#     from rpy2.robjects import pandas2ri
#     pandas2ri.activate()
#     r_dataframe = pandas2ri.conversion.py2rpy(df)
#
#     df_test = pd.DataFrame(test_x)
#     df_test.insert(loc=0, column='class', value=test_y + 1)
#     df_test.columns = ['class', 'X1', 'X2', 'X3']
#     r_dataframe_test = pandas2ri.conversion.py2rpy(df_test)
#
#     plot3ddb(nnew, robjects.IntVector(train_y + 1), r_dataframe, 'radial', gamma, coef0, cost, tolerance, probability_,
#              r_dataframe_test, True, title, filename)
#
#     # input('hello')


def plot_2D_decision_boundaries(
    auc,
    i,
    X_,
    y_,
    X_test,
    y_test,
    X_train,
    y_train,
    title,
    clf,
    folder,
    sub_dir_name,
    n_bin=8,
    save=True,
    DR="PCA",
):
    y_ = (y_ != 1).astype(int)
    y_test = (y_test != 1).astype(int)
    # print('graph...')
    # plt.subplots_adjust(top=0.75)
    # fig = plt.figure(figsize=(7, 6), dpi=100)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # plt.subplots_adjust(top=0.75)
    min = abs(X_.min()) + 1
    max = abs(X_.max()) + 1
    # print(X_lda.shape)
    # print(min, max)
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

    X_lda_0 = X_[y_ == 0]
    X_lda_1 = X_[y_ == 1]

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

    # X_lda_0_train = X_train[y_train == 0]
    # X_lda_1_train = X_train[y_train == 1]
    # ax.scatter(X_lda_0_train[:, 0], X_lda_0_train[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
    #            edgecolor="green", facecolors='none', label='Train data', zorder=1)
    #
    # ax.scatter(X_lda_1_train[:, 0], X_lda_1_train[:, 1], s=marker_size-10, vmin=-.2, vmax=1.2,
    #            edgecolor="green", facecolors='none', zorder=1)

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
    plt.title(title + " AUC=%.2f" % auc)
    ttl = ax.title
    ttl.set_position([0.57, 0.97])

    if save:
        path = "%s/decision_boundaries_graphs/%s/" % (folder, sub_dir_name)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = "%s_fold_%d.png" % (DR, i)
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


if __name__ == "__main__":
    dir_path = "F:/Data2/job_debug/ml"
    output_dir = "F:/Data2/job_debug/ml"
    build_roc_mosaic(dir_path, output_dir)
