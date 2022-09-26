import os
import pathlib
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from plotly.graph_objs.layout.scene import Annotation
from plotly.subplots import make_subplots
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme, element_text
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from tqdm import tqdm

from cwt._cwt import CWT, plot_line, STFT, plot_cwt_power, plot_stft_power, DWT
from utils.Utils import anscombe, concatenate_images
from utils._normalisation import CenterScaler

import umap.plot
import seaborn as sns
from collections import Counter
import matplotlib.cm as cm
from bokeh.plotting import figure, output_file, save
from highdimensional.decisionboundaryplot import DBPlot
from natsort import natsorted
import plotly.express as px

CSS_COLORS = {
    "QN_ANSCOMBE_LOG_STDS": "forestgreen",
    "LINEAR_QN_ANSCOMBE_LOG_CENTER_DWT": "teal",
    "LINEAR_QN_ANSCOMBE_LOG": "green",
    "LINEAR_QN_ANSCOMBE_LOG_CENTER_CWTMORL": "grey",
    "LINEAR_QN_ANSCOMBE_LOG_STD_APPEND_LINEAR_QN_ANSCOMBE_LOG_CENTER_CWTMORL": "brown",
    "LINEAR_QN_ANSCOMBE_LOG_STD_APPEND_LINEAR_QN_ANSCOMBE_LOG_CENTER_DWT": "coral",
    "LINEAR_QN_ANSCOMBE_LOG_STD_APPEND_LINEAR_QN_ANSCOMBE_CENTER_DWT": "pink",
    "LINEAR_QN_ANSCOMBE_CENTER_DWT": "olive",
    "LINEAR_QN_ANSCOMBE_LOG_STD": "black",
    "LINEAR_QN_STD": "cyan",
    "QN": "blue",
    "QN_STD": "navy",
    "QN_STD_CENTER_CWTMORL": "magenta",
    "QN_STD_CENTER_DWT": "indigo",
    "QN_ANSCOMBE": "purple",
    "QN_ANSCOMBE_LOG": "orange",
    "QN_ANSCOMBE_LOG_CENTER_CWTMORL": "red",
    "QN_ANSCOMBE_LOG_CENTER_DWT": "gold",
}


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
    df_healthy = df[df["health"] == 0].iloc[:, :-N_META].values
    df_unhealthy = df[df["health"] == 1].iloc[:, :-N_META].values

    assert len(df_healthy) > 0, "no healthy samples!"
    assert len(df_unhealthy) > 0, "no unhealthy samples!"

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

    df_healthy_ = add_separator(df_[df_["health"] == 0])
    df_unhealthy_ = add_separator(df_[df_["health"] == 1])

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


def plot_2d_space(
    X, y, filename_2d_scatter, label_series, title="title", colors=None, marker_size=4
):
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    print("plot_2d_space")
    if len(X[0]) == 1:
        for l in zip(np.unique(y)):
            ax.scatter(
                X[y == l, 0], np.zeros(X[y == l, 0].size), label=l, s=marker_size
            )
    else:
        for l in zip(np.unique(y)):
            if l[0] in label_series.keys():
                ax.scatter(
                    X[y == l[0]][:, 0],
                    X[y == l[0]][:, 1],
                    label=label_series[l[0]],
                    s=marker_size,
                )

    colormap = cm.get_cmap("Spectral")
    colorst = [colormap(i) for i in np.linspace(0, 0.9, len(ax.collections))]
    for t, j1 in enumerate(ax.collections):
        j1.set_color(colorst[t])
    ax.patch.set_facecolor("black")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=5)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    print(filename_2d_scatter)
    folder = Path(filename_2d_scatter).parent
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename_2d_scatter)
    # plt.show()
    plt.close(fig)
    plt.clf()


def plot_umap(meta_columns, df, output_dir, label_series, title="title", y_col="label"):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_before_reduction = df.iloc[:, : -len(meta_columns)].values
    mapper = umap.UMAP().fit(df_before_reduction)

    ids = df["id"].values
    labels = df[y_col].values
    seasons = (
        pd.to_datetime(df["date"], format="%d/%m/%Y").dt.month % 12 // 3 + 1
    ).map({1: "winter", 2: "spring", 3: "summer", 4: "fall"})
    filename = f"{title.replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(9.00, 9.00))
    umap.plot.points(mapper, labels=labels, ax=ax, background="black")
    filepath = output_dir / f"umap_plot_labels_{filename}"
    fig.savefig(filepath)
    print(filepath)

    fig, ax = plt.subplots(figsize=(9.00, 9.00))
    umap.plot.points(mapper, labels=ids, ax=ax, background="black")
    filepath = output_dir / f"umap_plot_ids_{filename}"
    fig.savefig(filepath)
    print(filepath)

    fig, ax = plt.subplots(figsize=(9.00, 9.00))
    umap.plot.points(mapper, labels=seasons, ax=ax, background="black")
    filepath = output_dir / f"umap_plot_seasons_{filename}"
    fig.savefig(filepath)
    print(filepath)

    # interactive umap
    hover_data = pd.DataFrame(
        {"index": np.arange(df_before_reduction.shape[0]), "label": ids}
    )
    # build meta list
    meta_list = []
    for index, row in df[meta_columns].iterrows():
        meta_str = ""
        for i, m in enumerate(meta_columns):
            if m == "id" or m == "target":
                continue
            meta_str += f"{m}={str(row[m])} "
        meta_list.append(meta_str)
    hover_data["item"] = meta_list
    print(hover_data)

    print("saving interactive umap...")
    p = umap.plot.interactive(
        mapper,
        labels=labels,
        hover_data=hover_data,
        point_size=5,
        background="black",
        interactive_text_search=True,
    )
    filepath = output_dir / f"umap_iplot_labels_{title.replace(' ', '_')}.html"
    output_file(str(filepath), mode="inline")
    save(p)
    print(filepath)

    print("saving interactive umap...")
    p = umap.plot.interactive(
        mapper,
        labels=ids,
        hover_data=hover_data,
        point_size=5,
        background="black",
        interactive_text_search=True,
    )
    filepath = output_dir / f"umap_iplot_ids_{title.replace(' ', '_')}.html"
    output_file(str(filepath), mode="inline")
    save(p)
    print(filepath)

    print("saving interactive umap...")
    p = umap.plot.interactive(
        mapper,
        labels=seasons,
        hover_data=hover_data,
        point_size=5,
        background="black",
        interactive_text_search=True,
    )
    filepath = output_dir / f"umap_iplot_seasons_{title.replace(' ', '_')}.html"
    output_file(str(filepath), mode="inline")
    save(p)
    print(filepath)


def plot_time_pca(
    meta_size, df, output_dir, label_series, title="title", y_col="label"
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame(PCA(n_components=2).fit_transform(df.iloc[:, :-meta_size])).values
    y = df["target"].astype(int)
    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = output_dir / filename
    plot_2d_space(X, y, filepath, label_series, title=title)


def plot_time_pls(
    meta_size, df, output_dir, label_series, title="title", y_col="label"
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    y = df["target"].astype(int)
    X = pd.DataFrame(
        PLSRegression(n_components=2).fit_transform(X=df.iloc[:, :-meta_size], y=y)[0]
    ).values

    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = output_dir / filename
    plot_2d_space(X, y, filepath, label_series, title=title)


def plot_time_lda(N_META, df, output_dir, label_series, title="title", y_col="label"):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
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
        config = [row["config"].replace("->",">").replace(" ","") for _ in range(len(test_balanced_accuracy_score))]
        data["test_balanced_accuracy_score"] = test_balanced_accuracy_score
        data["test_precision_score0"] = test_precision_score0
        data["test_precision_score1"] = test_precision_score1
        data["test_recall_score0"] = test_recall_score0
        data["test_recall_score1"] = test_recall_score1
        data["test_f1_score0"] = test_f1_score0
        data["test_f1_score1"] = test_f1_score1
        data["class0"] = row["class0"]
        data["class1"] = row["class1"]
        roc_auc_scores.extend(
            [0] * (len(test_balanced_accuracy_score) - len(roc_auc_scores))
        )  # in case auc could not be computed for fold
        data["roc_auc_scores"] = roc_auc_scores
        data["config"] = config
        dfs.append(data)
    formated = pd.concat(dfs, axis=0)
    return formated


def build_annotations(df, fig_auc_only):
    COLOR_MAP_IM = {
        1: "rgb(70,26,63)",
        2: "rgb(103,21,74)",
        3: "rgb(138,0,78)",
        4: "rgb(173,0,74)",
        5: "rgb(206,0,63)",
        6: "rgb(234,0,43)",
        7: "rgb(255,0,0)",
    }
    COLOR_MAP_AD = {
        1: "rgb(255,246,0)",
        2: "rgb(255,217,0)",
        3: "rgb(255,186,0)",
        4: "rgb(255,155,0)",
        5: "rgb(255,120,0)",
        6: "rgb(255,80,0)",
        7: "rgb(255,0,0)",
    }

    annotations = []
    for item in df["config"].unique():
        imputed_d = int(item.split("ID=")[1][0])
        activity_d = int(item.split("AD=")[1][0])
        proc_label = item.split(">")[-3]
        annot_str = f"{proc_label}"
        a = dict(
            x=item,
            y=df[df["config"] == item]["roc_auc_scores"].max(),
            text=annot_str,
            showarrow=False,
            # bgcolor=COLOR_MAP_AD[imputed_d],
            # font=dict(
            #     color=COLOR_MAP_IM[activity_d]
            #     # size=12
            # ),
        )
        annotations.append(a)
    return tuple(annotations)


def plot_ml_report_final(output_dir):
    print("building report visualisation...")
    dfs = []
    label_dict = {}
    paths = list(output_dir.glob("**/*.csv"))
    if len(paths) == 0:
        paths = list(output_dir.parent.glob("**/*.csv"))

    for path in paths:
        if "report" not in str(path):
            continue
        df = pd.read_csv(str(path), index_col=None)
        medians = []
        if "roc_auc_scores" not in df.columns:
            continue
        for value in df["roc_auc_scores"].values:
            v = stringArrayToArray(value)
            medians.append(np.median(v))
        df["median_auc"] = medians

        df["config"] = f"{df.steps[0]}{df.classifier[0]}"
        df = df.sort_values("median_auc")
        df = df.drop_duplicates(subset=["config"], keep="first")
        label_dict["UnHealthy"] = df["class1"].values[0]
        label_dict["Healthy"] = df["class0"].values[0]
        dfs.append(df)

    if len(dfs) == 0:
        print("no reports available.")
        return
    df = pd.concat(dfs, axis=0)
    df["health_tags"] = df["class_0_label"] + df["class_1_label"]
    df["color"] = [x.split(">")[-3] for x in df["config"].values]
    # df = df.sort_values(["median_auc", "color"], ascending=[True, True])
    for farm in df["farm_id"].unique():
        df_f = df[df["farm_id"] == farm]
        for h_tag in df_f["health_tags"].unique():
            df_f_ = df_f[df_f["health_tags"] == h_tag]

            df_f_ = df_f_.sort_values(["color", "median_auc"], ascending=[True, True])

            t4 = "AUC performance of different inputs<br>%s" % str(label_dict)

            t3 = "Accuracy performance of different inputs<br>%s" % str(label_dict)

            t1 = "Precision class0 performance of different inputs<br>%s" % str(
                label_dict
            )

            t2 = "Precision class1 performance of different inputs<br>%s" % str(
                label_dict
            )

            fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))
            fig_auc_only = make_subplots(rows=1, cols=1)

            df_f_ = formatForBoxPlot(df_f_)
            formated_label = []
            for label in df_f_['config'].values:
                split = label.split('>')
                label_formated = ""
                for i, item in enumerate(split):
                    label_formated += f"{item}>"
                    if i == len(split)-4:
                        label_formated += "<br>"
                formated_label.append(label_formated)
            df_f_['config'] = formated_label

            fig.append_trace(
                px.box(df_f_, x="config", y="test_precision_score0").data[0],
                row=1,
                col=1,
            )
            fig.append_trace(
                px.box(df_f_, x="config", y="test_precision_score1").data[0],
                row=2,
                col=1,
            )
            fig.append_trace(
                px.box(df_f_, x="config", y="test_balanced_accuracy_score").data[0],
                row=3,
                col=1,
            )
            fig.append_trace(
                px.box(df_f_, x="config", y="roc_auc_scores").data[0],
                row=4,
                col=1,
            )

            fig_auc_only.append_trace(
                px.box(df_f_, x="config", y="roc_auc_scores", title=t4).data[0],
                row=1,
                col=1,
            )
            #annot = build_annotations(df_f_, fig_auc_only)
            fig.update_xaxes(showticklabels=False)  # hide all the xticks
            fig.update_xaxes(showticklabels=True, row=4, col=1, automargin=True)

            fig.update_yaxes(showgrid=True, gridwidth=1, automargin=True)
            fig.update_xaxes(showgrid=True, gridwidth=1, automargin=True)
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=500))
            fig_auc_only.update_layout(margin=dict(l=20, r=20, t=20, b=500))
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"ML_performance_final_{farm}.html"
            print(filepath)
            fig.write_html(str(filepath))
            filepath = output_dir / f"ML_performance_final_auc_{farm}.html"
            print(filepath)
            # fig_auc_only.write_html(str(filepath))
            # fig.show()

            x_data = df_f_["config"].unique()

            color_data = [x.split(">")[-4] for x in x_data]
            imp_days_data = [x.split(">")[1].split('=')[1] for x in x_data]
            y_data = []
            for x in df_f_["config"].unique():
                y_data.append(df_f_[df_f_["config"] == x]["roc_auc_scores"].values)
            traces = []
            colors = []
            class0_list = []
            class1_list = []
            sec_axis = []
            for i_d, c, xd, yd in zip(imp_days_data, color_data, x_data, y_data):
                class0 = df_f_[df_f_["config"] == xd]["class0"].unique()
                class1 = df_f_[df_f_["config"] == xd]["class1"].unique()
                imp_days = df_f_[df_f_["config"] == xd]["class1"].unique()
                class0_list.append(class0)
                class1_list.append(class1)
                color = CSS_COLORS[c.replace("(", '').replace(")", '')]
                colors.append(color)
                traces.append(
                    go.Bar(
                        y=class0,
                        x=[xd],
                        name="Healthy samples",
                        width=[0.25],
                        offsetgroup="Healthy samples",
                        marker=dict(color="#1f77b4"),
                        opacity=0.2,
                        showlegend=False
                    )
                )
                sec_axis.append(False)
                traces.append(
                    go.Bar(
                        y=class1,
                        x=[xd],
                        name="Unhealthy samples",
                        width=[0.25],
                        offsetgroup="Unhealthy samples",
                        marker=dict(color="#ff7f0e"),
                        opacity=0.2,
                        showlegend=False
                    )
                )
                sec_axis.append(False)
                # traces.append(
                #     go.Bar(
                #         y=[i_d],
                #         x=[xd],
                #         name="Imputed days",
                #         width=[0.25],
                #         offsetgroup="Imputed days",
                #         marker=dict(color="#7f7f7f"),
                #         opacity=0.3,
                #     )
                # )
                # sec_axis.append(False)
                traces.append(
                    go.Box(
                        y=yd,
                        name=xd,
                        boxpoints="outliers",
                        marker=dict(color=color, size=10),
                        legendgroup=c,
                        marker_color=color,
                        marker_size=2,
                        line_width=1 if float(i_d) < 0 else float(i_d)*0.5,
                        showlegend=False
                    )
                )
                sec_axis.append(True)

            for c in np.unique(color_data):
                traces.append(
                    go.Box(
                        y=yd,
                        name=c,
                        boxpoints="outliers",
                        marker=dict(color=CSS_COLORS[c.replace("(", '').replace(")", '')], size=10),
                        marker_color=CSS_COLORS[c.replace("(", '').replace(")", '')],
                        showlegend = True,
                    )
                )
                sec_axis.append(True)

            traces.append(
                go.Bar(
                    y=class0,
                    x=[xd],
                    name="Healthy samples",
                    width=[0],
                    offsetgroup="Healthy samples",
                    marker=dict(color="#1f77b4"),
                    opacity=0.8,
                    showlegend=True
                )
            )
            sec_axis.append(False)
            traces.append(go.Bar(
                y=class0,
                x=[xd],
                name="Unhealthy samples",
                width=[0],
                offsetgroup="Unhealthy samples",
                marker=dict(color="#ff7f0e"),
                opacity=0.8,
                showlegend=True
            ))
            sec_axis.append(False)

            h_labels = df_f_["config"].values[0].split(">H=")[1].split(">")[0]
            uh_labels = df_f_["config"].values[0].split(">UH=")[1].split(">")[0]
            # fig_ = go.Figure(data=traces)
            fig_ = make_subplots(specs=[[{"secondary_y": True}]])
            for a, t in zip(sec_axis, traces):
                fig_.add_trace(t, secondary_y=a)
                # fig_.add_trace(go.Bar(name=t.name, y=t.y), secondary_y=True)
                # fig_.add_trace(go.Bar(name=t.name, y=t.y), secondary_y=True)

            # for t in [
            #     go.Bar(name='Healthy samples', y=[100]),
            #     go.Bar(name='Unhealthy samples', y=[100])
            # ]:
            #     fig_.add_trace(t, secondary_y=True)

            fig_.update_yaxes(showgrid=True, gridwidth=1, automargin=True)
            fig_.update_layout(
                title=f"healthy labels={h_labels} unhealthy labels={uh_labels}",
                yaxis_title="AUC"
            )
            fig_.update_xaxes(tickangle=-45)

            # d = {}
            # for i, trace in enumerate(fig_["data"]):
            #     if trace.marker["color"] in d.keys():
            #         trace["showlegend"] = False
            #     else:
            #         # name = trace["name"].split(">")
            #         # if len(name) > 1:
            #         #     name = name[-3]
            #         # else:
            #         #     name = name[0]
            #         # trace["name"] = name
            #         trace["showlegend"] = True
            #     d[trace.marker["color"]] = i

            filepath = output_dir / f"ML_performance_final_auc_{farm}_{h_tag}.html"
            print(filepath)
            fig_.update_layout(barmode='group')
            fig_.update_yaxes(title_text="AUC", secondary_y=True)
            fig_.update_yaxes(title_text="Sample count", secondary_y=False)
            fig_.update_xaxes(range=[-1, len(x_data)-0.5])
            fig_.write_html(str(filepath))


def plot_ml_report(clf_name, path, output_dir):
    print("building report visualisation...")
    df = pd.read_csv(str(path), index_col=None)
    medians = []
    for value in df["roc_auc_scores"].values:
        v = stringArrayToArray(value)
        medians.append(np.median(v))
    df["median_auc"] = medians

    df["config"] = f"{df.steps[0]}{df.classifier[0]}"
    df = df.sort_values("median_auc")
    df = df.drop_duplicates(subset=["config"], keep="first")
    df = df.fillna(-1)
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
    filepath = output_dir / f"ML_performance_{clf_name}.html"
    print(filepath)
    fig.write_html(str(filepath))
    # fig.show()


def plot_zeros_distrib(
    meta_columns,
    a_days,
    label_series,
    data_frame_no_norm,
    graph_outputdir,
    title="Percentage of zeros in activity per sample",
):
    if a_days is None:
        print("skip plot_zeros_distrib.")
        return
    print("plot_zeros_distrib...")
    data = {}
    target_labels = []
    z_prct = []

    for index, row in data_frame_no_norm.iterrows():
        a = row[: -len(meta_columns)].values
        label = label_series[row["target"]]

        target_labels.append(label)
        z_prct.append(np.sum(a == 0) / len(a))

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
        + theme(
            subplots_adjust={"right": 0.82}, axis_text_x=element_text(angle=90, hjust=1)
        )
    )

    fig = g.draw()
    fig.tight_layout()
    # fig.show()
    filename = f"zero_percent_{title.lower().replace(' ', '_')}.png"
    filepath = graph_outputdir / filename
    # print('saving fig...')
    print(filepath)
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


def plot_all_features(
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
    xaxis="Time",
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
    ax_roc_merge,
    ax,
    tprs_test,
    mean_fpr_test,
    aucs_test,
    tprs_train,
    mean_fpr_train,
    aucs_train,
    out_dir,
    classifier_name,
    fig,
    fig_roc_merge,
    cv_name,
    days,
    info="None",
    tag="",
    export_fig_as_pdf=False
):
    ax_roc_merge.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )
    ax[0].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )
    ax[1].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0
    mean_auc_test = auc(mean_fpr_test, mean_tpr_test)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs_test)

    label = f"Mean ROC Test (Median AUC = {np.median(aucs_test):.2f}, 95% CI [{lo:.4f}, {hi:.4f}] )"
    if len(aucs_test) <= 2:
        label = r"Mean ROC (Median AUC = %0.2f)" % np.median(aucs_test)
    ax[1].plot(mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1)

    ax[1].set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Testing data) Receiver operating characteristic days:{days} cv:{cv_name} \n info:{info}",
    )
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")
    ax[1].legend(loc="lower right")

    ax_roc_merge.plot(
        mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training/Testing data) Receiver operating characteristic days:{days} cv:{cv_name} \n info:{info}",
    )
    ax_roc_merge.set_xlabel("False positive rate")
    ax_roc_merge.set_ylabel("True positive rate")
    ax_roc_merge.legend(loc="lower right")
    # fig.show()

    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr_train, mean_tpr_train)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs_train)

    label = f"Mean ROC Training (Median AUC = {np.median(aucs_train):.2f}, 95% CI [{lo:.4f}, {hi:.4f}] )"
    if len(aucs_train) <= 2:
        label = r"Mean ROC (Median AUC = %0.2f)" % np.median(aucs_train)
    ax[0].plot(
        mean_fpr_train, mean_tpr_train, color="black", label=label, lw=2, alpha=1
    )

    ax[0].set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training data) Receiver operating characteristic days:{days} cv:{cv_name} \n info:{info}",
    )
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].legend(loc="lower right")

    ax_roc_merge.plot(
        mean_fpr_train, mean_tpr_train, color="red", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training data) Receiver operating characteristic days:{days} cv:{cv_name} \n info:{info}",
    )
    ax_roc_merge.legend(loc="lower right")

    fig.tight_layout()
    path = out_dir / "roc_curve" / cv_name
    path.mkdir(parents=True, exist_ok=True)
    # final_path = path / f"{tag}_roc_{classifier_name}.png"
    # print(final_path)
    # fig.savefig(final_path)

    final_path = path / f"{tag}_roc_{classifier_name}_merge.png"
    print(final_path)
    fig_roc_merge.savefig(final_path)

    filepath = out_dir.parent / f"{out_dir.stem}_{tag}_roc_{classifier_name}_merge.png"
    print(filepath)
    fig_roc_merge.savefig(filepath)

    if export_fig_as_pdf:
        final_path = path / f"{tag}_roc_{classifier_name}_merge.pdf"
        print(final_path)
        fig_roc_merge.savefig(final_path)

        filepath = out_dir.parent / f"{out_dir.stem}_{tag}_roc_{classifier_name}_merge.pdf"
        print(filepath)
        fig_roc_merge.savefig(filepath)


    # filepath = out_dir.parent / f"{out_dir.stem}_{tag}_roc_{classifier_name}.png"
    # print(filepath)
    # fig.savefig(filepath)

    # path = "%s/roc_curve/svg/" % out_dir
    # create_rec_dir(path)
    # final_path = '%s/%s' % (path, 'roc_%s.svg' % classifier_name)
    # print(final_path)
    # fig.savefig(final_path)
    return mean_auc_test


def plot_distribution(X, output_dir, filename):
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


# def figures_to_html(figs, out_dir, ncol=2, maxrow=15):
#     fig_list = chunks(figs, maxrow*2)
#
#     for i, ff in enumerate(fig_list):
#         fig = make_subplots(rows=maxrow, cols=ncol)
#         row = 1
#         col = 1
#         for f in ff:
#             fig.add_trace(f["data"][0], row=row, col=col)
#             #print(row, col)
#             col += 1
#             if col > ncol:
#                 col = 1
#                 row += 1
#
#         fig.update_layout(
#             autosize=False,
#             width=1200,
#             height=1700
#         )
#         filename = out_dir / f"dashboard_{i}.html"
#         fig.write_html(str(filename))
#
#     # dashboard = open(filename, "w")
#     # dashboard.write("<html><head></head><body>" + "\n")
#     # for i, fig in enumerate(figs):
#     #
#     #     if i % 3 == 0:
#     #         dashboard.write("<div class='row'>" + "\n")
#     #
#     #     dashboard.write("\t<div class='column'>" + "\n")
#     #
#     #     inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
#     #     dashboard.write(inner_html)
#     #
#     #     dashboard.write("\t</div>" + "\n")
#     #
#     #     if i % 3 == 0:
#     #         dashboard.write("</div>" + "\n")
#     #
#     # dashboard.write("</body></html>" + "\n")
#


def rolling_window(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]


def plot_mean_groups(
    sub_sample_scales,
    n_scales,
    sfft_window,
    wavelet_f0,
    dwt_w,
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
        if df_.shape[0] == 0:
            continue
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

        try:
            dfs_mean = [(g['name'].values[0], np.mean(g.iloc[:, :-N_META], axis=0)) for _, g in df_.groupby(['name'])]
            dfs_median = [(g['name'].values[0], np.median(g.iloc[:, :-N_META], axis=0)) for _, g in df_.groupby(['name'])]

            for m1, m2 in zip(dfs_mean, dfs_median):
                fig_group.add_trace(
                    go.Scatter(x=x, y=m1[1].values, mode="lines", name=f"Mean {m1[0]}", line_color="#000000")
                )
                # fig_group.add_trace(
                #     go.Scatter(x=x, y=m2[1].values, mode="lines", name=f"Median {m2[0]}", line_color="#000000")
                # )
                fig_group_means.add_trace(
                    go.Scatter(x=x, y=m1[1].values, mode="lines", name=f"Mean {m1[0]}")
                )
                fig_group_median.add_trace(
                    go.Scatter(x=x, y=m2[1], mode="lines", name=f"Mean {m2[0]}")
                )

        except Exception as e:
            print(e)

        s = mean.values
        s = anscombe(s)
        s = np.log(s)

        plot_line(
            np.array([s]),
            out_dir,
            label + "_" + str(df_.shape[0]),
            label + "_" + str(df_.shape[0]) + ".html",
        )
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
                vmin=0,
                vmax=3,
                enable_graph_out=True,
                sub_sample_scales=sub_sample_scales,
            ).transform([s])

        #if sfft_window is not None:
        STFT(
            enable_graph_out=True,
            sfft_window=sfft_window,
            out_dir=out_dir,
            step_slug="ANSCOMBE_" + label + "_" + str(df_.shape[0]),
            animal_ids=[],
            targets=[],
            dates=[],
        ).transform([s])

        #if dwt_w is not None:
        DWT(
            enable_graph_out = True,
            dwt_window=dwt_w,
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
            xaxis_title="Time",
            yaxis_title="Activity (count)",
        )
        fig_group_means.update_layout(
            title="Mean of samples for each category",
            xaxis_title="Time",
            yaxis_title="Activity (count)",
        )
        fig_group_median.update_layout(
            title="Median of samples for each category",
            xaxis_title="Time",
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
    # stack cwt figures
    files = list((out_dir / "_cwt").glob("*.png"))
    concatenate_images(files, out_dir)


def plot_mosaic(output_dir, cv_name, directory_t, filename, subdir):
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
#


def plot_high_dimension_db(out_dir, X, y, train_index, meta, clf, days, steps, ifold, export_fig_as_pdf):
    """
    Plot high-dimensional decision boundary
    """
    print(f"plot_high_dimension_db {ifold}")
    try:
        # db = DBPlot(clf)
        # db.fit(X, y, training_indices=train_index)
        # fig, ax = plt.subplots(figsize=(19.20, 19.20))
        # db.plot(
        #     ax, generate_testpoints=True, meta=meta
        # )  # set generate_testpoints=False to speed up plotting
        # models_visu_dir = (
        #     out_dir
        #     / "models_visu_pca"
        #     / f"{type(clf).__name__}_{clf.kernel}_{days}_{steps}"
        # )
        # models_visu_dir.mkdir(parents=True, exist_ok=True)
        # filepath = models_visu_dir / f"{ifold}.png"
        # print(filepath)
        # plt.savefig(filepath)
        # plot_learning_curves(clf, X, y, ifold, models_visu_dir)

        db = DBPlot(clf, dimensionality_reduction=PLSRegression(n_components=2))
        db.fit(X, y, training_indices=train_index)
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        db.plot(
            ax, generate_testpoints=False, meta=meta
        )  # set generate_testpoints=False to speed up plotting
        models_visu_dir = (
            out_dir
            / "models_visu_pls"
            / f"{type(clf).__name__}_{clf.kernel}_{days}_{steps}"
        )
        models_visu_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_visu_dir / f"{ifold}.png"
        print(filepath)
        plt.savefig(filepath)
        if export_fig_as_pdf:
            filepath = models_visu_dir / f"{ifold}.pdf"
            print(filepath)
            plt.savefig(filepath)

        # plot_learning_curves(clf, X, y, ifold, models_visu_dir)
    except Exception as e:
        print(e)


def plot_learning_curves(clf, X, y, ifold, models_visu_dir):
    # plot learning curves for comparison
    fig, ax = plt.subplots()
    N = 10
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5)
    ax.errorbar(
        train_sizes,
        np.mean(train_scores, axis=1),
        np.std(train_scores, axis=1) / np.sqrt(N),
    )
    ax.errorbar(
        train_sizes,
        np.mean(test_scores, axis=1),
        np.std(test_scores, axis=1) / np.sqrt(N),
        c="r",
    )

    ax.legend(["Accuracies on training set", "Accuracies on test set"])
    ax.set_xlabel("Number of data points")
    ax.set_title(str(clf))
    models_visu_dir.mkdir(parents=True, exist_ok=True)
    filepath = (
        models_visu_dir
        / f"learning_curve_{ifold}_{type(clf).__name__}_{clf.kernel}.png"
    )
    print(filepath)
    plt.savefig(filepath)


def plot_2d_decision_boundaries(
    auc,
    i,
    X,
    y,
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
    dimensionality_reduction="PCA",
):
    y = (y != 1).astype(int)
    y_test = (y_test != 1).astype(int)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    min = abs(X.min()) + 1
    max = abs(X.max()) + 1
    # print(X_lda.shape)
    # print(min, max)
    # if np.max([min, max]) > 100:
    #     return
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

    X_0 = X[y == 0]
    X_1 = X[y == 1]

    X_0_t = X_test[y_test == 0]
    X_1_t = X_test[y_test == 1]
    marker_size = 150
    ax.scatter(
        X_0[:, 0],
        X_0[:, 1],
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
        X_1[:, 0],
        X_1[:, 1],
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
        X_0_t[:, 0],
        X_0_t[:, 1],
        s=marker_size - 10,
        vmin=-0.2,
        vmax=1.2,
        edgecolor="black",
        facecolors="none",
        label="Test data",
        zorder=1,
    )

    ax.scatter(
        X_1_t[:, 0],
        X_1_t[:, 1],
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

    path = folder / "decision_boundaries_graphs" / sub_dir_name
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = f"{dimensionality_reduction}_fold_{i}.png"
    final_path = path / filename
    print(final_path)
    plt.savefig(str(final_path), bbox_inches="tight")


def build_roc_curve(output_dir, label_unhealthy, scores):
    print("build_roc_curve...")


def plot_fold_details(
    fold_results, meta, meta_columns, out_dir, filename="fold_details"
):
    #print(fold_results)
    #create one histogram per test fold (for loo)
    try:
        hist_list = []
        names = []
        for item in fold_results:
            probs = [x[1] for x in item['y_pred_proba_test']]
            test_fold_name = f"{item['meta_test'][0][7]}_{item['meta_test'][0][1]}_{item['meta_test'][0][0]}"
            names.append(test_fold_name)
            plt.clf()
            h, _, _ = plt.hist(probs, density=True, bins=50, alpha=0.5, label=f"prob of sample (mean={np.mean([x[1] for x in item['y_pred_proba_test']]):.2f})")
            hist_list.append(h)
            plt.ylabel("Density")
            plt.xlabel("Probability of being unhealthy(target=1) per sample(perm of peaks)")
            plt.xlim(xmin=0, xmax=1)
            plt.title(
                f"Histograms of prediction probabilities\n{test_fold_name} testing_shape={item['testing_shape']} target={item['meta_test'][0][0]}")
            plt.axvline(x=0.5, color="gray", ls="--")
            plt.legend(loc="upper right")
            # plt.show()
            filename = f"histogram_of_prob_{test_fold_name}.png"
            out = out_dir / 'loo_histograms'
            out.mkdir(parents=True, exist_ok=True)
            filepath = out / filename
            print(filepath)
            plt.savefig(str(filepath))

        hist_list = np.array(hist_list)
        fig, ax = plt.subplots(figsize=(8.20, 7.20))
        im = ax.imshow(hist_list)

        x_axis = [f"{x:.1f}" for x in np.linspace(start=0, stop=1, num=hist_list.shape[1])]
        ax.set_xticks(np.arange(len(x_axis)), labels=x_axis)
        ax.set_yticks(np.arange(len(names)), labels=names)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        ax.set_title("Histograms of test prediction probabilities")

        filename = f"heatmap_histogram_of_prob.png"
        out = out_dir / 'loo_histograms'
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / filename

        fig.tight_layout()
        print(filepath)
        fig.savefig(filepath)

    except Exception as e:
        print(e)

    meta_dict = {}
    for m in meta:
        id = 0
        if "id" in meta_columns:
            id = m[meta_columns.index("id")]
        target = 0
        if "target" in meta_columns:
            target = str(m[meta_columns.index("target")])
        name = 0
        if "name" in meta_columns:
            name = m[meta_columns.index("name")]
        meta_dict[id] = f"{target} {name}"
    data = []
    for f in fold_results:
        i_fold = f["i_fold"]
        accuracy_train = f["accuracy_train"]
        accuracy = f["accuracy"]

        ids_test = np.unique(f["ids_test"]).astype(float)
        ids_train = np.unique(f["ids_train"]).astype(float)

        ids_test_ = np.vectorize(meta_dict.get)(ids_test)
        ids_train_ = np.vectorize(meta_dict.get)(ids_train)

        data.append([accuracy_train, accuracy, ids_test_, ids_train_])

    df = pd.DataFrame(data, columns=["accuracy_train", "accuracy_test", "ids_test", "ids_train"])
    mean_acc_train = np.mean(df["accuracy_train"].values)
    mean_acc = np.mean(df["accuracy_test"].values)
    df = df.sort_values(by="accuracy_test")
    filepath = out_dir / f"{filename}.csv"
    df.to_csv(filepath, index=False)

    df_test = df[["accuracy_test", "accuracy_train", "ids_test"]]
    df_test = df_test.sort_values(by="accuracy_test")
    df_test.index = df["ids_test"]
    ax = df_test.plot.bar(
        rot=90,
        log=False,
        figsize=(0.8 * len(fold_results), 7.20),
        title=f"Classifier predictions per fold n={len(fold_results)} mean_acc_train={mean_acc_train:.2f} mean_acc_test={mean_acc:.2f}",
    )
    ax.axhline(y=0.5, color='r', linestyle='--')
    try:
        for item in ax.get_xticklabels():
            if int(item.get_text().split(' ')[0].replace('[', '')) == 0:
                item.set_color("tab:blue")
    except ValueError as e:
        print(e)

    ax.set_xlabel("Fold metadata")
    ax.set_ylabel("Accuracy")
    fig = ax.get_figure()
    filepath = out_dir / f"{filename}_test.png"
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    # plt.clf()
    # df_train = df[["auc_train", "ids_train"]]

    # df_train = df_train.sort_values(by='auc_train')
    # df_train.index = df_train["ids_train"]
    # ax_train = df_train.plot.bar(
    #     rot=90,
    #     log=False,
    #     figsize=(0.3*len(fold_results), 30.20),
    #     title=f"(Training) Classifier predictions per fold n={len(fold_results)}",
    # )
    # ax_train.set_xlabel("Fold metadata")
    # ax_train.set_ylabel("AUC")
    # fig_train = ax_train.get_figure()
    # filepath = out_dir / f"{filename}_training.png"
    # print(filepath)
    # #fig_train.tight_layout()
    # fig_train.savefig(filepath)


def build_individual_animal_pred(
    output_dir, steps, label_unhealthy, scores, ids, meta_columns, tt="test"
):
    print("build_individual_animal_pred...")
    for k, v in scores.items():
        # prepare data holder
        data_c_, data_c, data_i, data_c_prob, data_i_prob, data_m = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        for id in ids:
            data_c[id] = 0
            data_i[id] = 0
            data_c_[id] = []
            data_c_prob[id] = []
            data_i_prob[id] = []
            d = {}
            for m in meta_columns:
                d[m] = []
            data_m[id] = d

        score = scores[k]
        (
            data_dates,
            data_corr,
            data_incorr,
            prob_corr,
            prob_incorr,
            data_ids,
            data_meta,
        ) = ([], [], [], [], [], [], [])
        for s in score:
            dates = pd.to_datetime(s[f"sample_dates_{tt}"]).tolist()
            correct_predictions = s[f"correct_predictions_{tt}"]
            incorrect_predictions = s[f"incorrect_predictions_{tt}"]
            y_pred_proba_1 = np.array(s[f"y_pred_proba_{tt}"])[:, 1]
            y_pred_proba_0 = np.array(s[f"y_pred_proba_{tt}"])[:, 0]
            ids_test = s[f"ids_{tt}"]
            meta_test = s[f"meta_{tt}"]

            data_dates.extend(dates)
            data_corr.extend(correct_predictions)
            data_incorr.extend(incorrect_predictions)
            prob_corr.extend(y_pred_proba_0)
            prob_incorr.extend(y_pred_proba_1)
            data_ids.extend(ids_test)
            data_meta.extend(meta_test)

            for i in range(len(ids_test)):

                for j, m in enumerate(meta_columns):
                    data_m[ids_test[i]][m].append(meta_test[i][j])

                data_c[ids_test[i]] += correct_predictions[i]
                data_i[ids_test[i]] += incorrect_predictions[i]
                data_c_[ids_test[i]].append(correct_predictions[i])
                data_c_prob[ids_test[i]].append(y_pred_proba_0[i])
                data_i_prob[ids_test[i]].append(y_pred_proba_1[i])

        labels = list(data_c.keys())

        # labels_new = []
        # for l in labels:
        #     if "name" in data_m[l] and "age" in data_m[l]:
        #         print(l, data_m[l]["name"])
        #         name = data_m[l]["name"][0]
        #         age = str(data_m[l]["age"][0]).zfill(6)
        #         labels_new.append(f"{age} {name}")
        # if len(labels_new) > 0:
        #     labels = labels_new

        correct_pred = list(data_c.values())
        incorrect_pred = list(data_i.values())
        correct_pred_prob = list(data_c_prob.values())
        incorrect_pred_prob = list(data_i_prob.values())
        meta_pred = list(data_m.values())
        # make table
        df_table = pd.DataFrame(meta_pred, index=labels)
        for m in meta_columns:
            df_table[m] = [str(dict(Counter(x))) for x in df_table[m]]

        df_table["individual id"] = labels
        df_table["correct prediction"] = correct_pred
        df_table["incorrect prediction"] = incorrect_pred
        df_table["correct prediction_prob"] = correct_pred_prob
        df_table["incorrect prediction_prob"] = incorrect_pred_prob
        df_table["ratio of correct prediction (percent)"] = (
            df_table["correct prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        filename = f"table_data_{tt}_{k}.csv"
        filepath = output_dir / filename
        print(filepath)
        df_table = df_table.sort_values(
            "ratio of correct prediction (percent)", ascending=False
        )
        df_table.to_csv(filepath, index=False)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df_table.columns),
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=df_table.transpose().values.tolist(),
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        filename = f"table_data_{tt}_{k}.html"
        filepath = output_dir / filename
        print(filepath)
        fig.write_html(str(filepath))

        # print box plot
        plt.clf()
        max_v = max([len(x) for x in correct_pred_prob])
        correct_pred_prob_fix = []
        for item in correct_pred_prob:
            item += [np.nan] * (max_v - len(item))
            correct_pred_prob_fix.append(item)

        df = pd.DataFrame(
            {"correct_prediction_prob": correct_pred_prob_fix}, index=labels
        ).T
        df_c_p = df.apply(lambda x: x.explode() if x.name in df.columns else x)
        df_c_p = df_c_p.apply(lambda x: x.explode() if x.name in df.columns else x)
        df_c_p = df_c_p.reset_index(drop=True)
        # df_ = pd.concat([df_c_p, df_i_p], axis=1)
        df_ = df_c_p
        # df_ = df_.reindex(natsorted(df_.columns), axis=1)
        df_ = df_.astype(float)
        fig_ = plt.figure()
        boxplot = df_.boxplot(column=list(df_.columns), rot=90, figsize=(19.20, 10.80))
        boxplot.set_ylim(ymin=0, ymax=1)
        boxplot.axhline(y=0.5, color="gray", linestyle="--")
        boxplot.set_title(
            f"Classifier predictions probability ({tt}) \n per individual label_unhealthy={label_unhealthy}"
        )
        boxplot.set_xlabel("Individual")
        boxplot.set_ylabel("Probability")
        vals, names, xs = [], [], []
        for i, col in enumerate(df_.columns):
            vals.append(df_[col].values)
            names.append(col)
            xs.append(np.random.normal(i + 1, 0.04, df_[col].values.shape[0]))
        for n, (x, val) in enumerate(zip(xs, vals)):
            scatter = boxplot.scatter(
                x,
                val,
                alpha=1,
                marker="o",
                s=15,
                facecolors="none",
                edgecolors=[
                    "tab:blue" if x == 1 else "tab:red"
                    for x in list(data_c_.values())[n]
                ],
            )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Correct prediction",
                markeredgecolor="tab:blue",
                markerfacecolor="none",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Incorrect prediction",
                markeredgecolor="tab:red",
                markerfacecolor="none",
                markersize=5,
            ),
        ]
        boxplot.legend(handles=legend_elements, loc="lower right")

        filepath = output_dir / f"predictions_per_individual_box_{k}_{steps}_{tt}.png"
        print(filepath)
        fig_.tight_layout()
        fig_.savefig(filepath)

        # print figure
        plt.clf()
        df = pd.DataFrame(
            {
                "correct prediction": correct_pred,
                "incorrect prediction": incorrect_pred,
            },
            index=labels,
        )
        df = df.sort_index()
        df = df.astype(np.double)
        df = df.sort_values("correct prediction")
        df["correct prediction"] = (
            df_table["correct prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        df["incorrect prediction"] = (
            df_table["incorrect prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        ax = df.plot.bar(
            rot=90,
            log=False,
            title=f"Classifier predictions ({tt}) per individual label_unhealthy={label_unhealthy}",
        )
        ax.set_xlabel("Animals")
        ax.set_ylabel("Number of predictions")
        fig = ax.get_figure()
        filepath = output_dir / f"predictions_per_individual_{k}_{steps}_{tt}.png"

        # IDS = [
        #     "Greg",
        #     "Henry",
        #     "Tilly",
        #     "Maisie",
        #     "Sookie",
        #     "Oliver_F",
        #     "Ra",
        #     "Hector",
        #     "Jimmy",
        #     "MrDudley",
        #     "Kira",
        #     "Lucy",
        #     "Louis",
        #     "Luna_M",
        #     "Wookey",
        #     "Logan",
        #     "Ruby",
        #     "Kobe",
        #     "Saffy_J",
        #     "Enzo",
        #     "Milo",
        #     "Luna_F",
        #     "Oscar",
        #     "Kia",
        #     "Cat",
        #     "AlfieTickles",
        #     "Phoebe",
        #     "Harvey",
        #     "Mia",
        #     "Amadeus",
        #     "Marley",
        #     "Loulou",
        #     "Bumble",
        #     "Skittle",
        #     "Charlie_O",
        #     "Ginger",
        #     "Hugo_M",
        #     "Flip",
        #     "Guinness",
        #     "Chloe",
        #     "Bobby",
        #     "QueenPurr",
        #     "Jinx",
        #     "Charlie_B",
        #     "Thomas",
        #     "Sam",
        #     "Max",
        #     "Oliver_S",
        #     "Millie",
        #     "Clover",
        #     "Bobbie",
        #     "Gregory",
        #     "Kiki",
        #     "Hugo_R",
        #     "Shadow",
        # ]
        # COLOR_MAP = {}
        # cm = plt.get_cmap("gist_rainbow")
        # for i, c in enumerate(IDS):
        #     COLOR_MAP[c] = cm(i // 3 * 3.0 / len(IDS))
        # for xtick in ax.get_xticklabels():
        #     label = str(xtick).split(" ")[-1].replace("'", "").replace(")", "")
        #     print(label, COLOR_MAP[label])
        #     xtick.set_color(COLOR_MAP[label])

        print(filepath)
        fig.tight_layout()
        fig.savefig(filepath)

        # figure with time
        plt.clf()
        df = pd.DataFrame(
            {
                "data_dates": data_dates,
                "data_corr": data_corr,
                "data_ids": data_ids,
                "prob_corr": prob_corr,
            }
        )
        df = df.sort_values(by="data_dates")
        dfs = [group for _, group in df.groupby(df["data_dates"].dt.strftime("%B/%Y"))]
        dfs = sorted(dfs, key=lambda x: x["data_dates"].max(axis=0))
        fig, axs = plt.subplots(
            3, int(np.ceil(len(dfs) / 3)), facecolor="white", figsize=(28.0, 10.80)
        )
        fig.suptitle(
            f"Classifier predictions ({tt}) per individual across study time label_unhealthy={label_unhealthy}",
            fontsize=14,
        )
        axs = axs.ravel()

        fig_, axs_ = plt.subplots(
            3, int(np.ceil(len(dfs) / 3)), facecolor="white", figsize=(28.0, 10.80)
        )
        fig_.suptitle(
            f"Classifier predictions probability({tt}) per individual across study time label_unhealthy={label_unhealthy}",
            fontsize=14,
        )
        axs_ = axs_.ravel()

        for ax in axs:
            ax.set_axis_off()
        for ax in axs_:
            ax.set_axis_off()

        for i, d in enumerate(dfs):
            data_c, data_u, data_c_proba, data_u_proba = {}, {}, {}, {}
            for id in ids:
                data_c[id] = 0
                data_u[id] = 0
                data_c_proba[id] = []
                data_u_proba[id] = []
            for index, row in d.iterrows():
                data_c_proba[row["data_ids"]].append(row["prob_corr"])
                if row["data_corr"] == 1:
                    data_c[row["data_ids"]] += 1
                else:
                    data_u[row["data_ids"]] += 1
            labels = list(data_c.keys())
            correct_pred = list(data_c.values())
            incorrect_pred = list(data_u.values())
            correct_pred_prob = list(data_c_proba.values())
            df = pd.DataFrame(
                {
                    "correct prediction": correct_pred,
                    "incorrect prediction": incorrect_pred,
                },
                index=labels,
            )
            df.plot.bar(
                ax=axs[i],
                rot=90,
                log=False,
                title=pd.to_datetime(d["data_dates"].values[0]).strftime("%B %Y"),
            )
            axs[i].set_ylabel("Number of predictions")
            axs[i].set_xlabel("Individual")
            axs[i].set_axis_on()
            ######################
            max_v = max([len(x) for x in correct_pred_prob])
            correct_pred_prob_fix = []
            for item in correct_pred_prob:
                item += [np.nan] * (max_v - len(item))
                correct_pred_prob_fix.append(item)

            df_c = pd.DataFrame(
                {"correct_prediction_prob": correct_pred_prob_fix}, index=labels
            ).T

            df_c_p = df_c.apply(lambda x: x.explode() if x.name in df_c.columns else x)
            df_c_p = df_c_p.apply(
                lambda x: x.explode() if x.name in df_c.columns else x
            )

            # df_c_p = df_c.explode(list(df_c.columns))
            df_c_p = df_c_p.reset_index(drop=True)
            df_ = df_c_p
            # df_ = df_.reindex(natsorted(df_.columns), axis=1)
            df_ = df_.astype(float)
            boxplot = df_.boxplot(
                column=list(df_.columns), ax=axs_[i], rot=90, figsize=(12.80, 7.20)
            )
            axs_[i].set_title(
                pd.to_datetime(d["data_dates"].values[0]).strftime("%B %Y")
            ),
            axs_[i].axhline(y=0.5, color="gray", linestyle="--")
            axs_[i].set_ylim(ymin=0, ymax=1)
            axs_[i].set_xlabel("Individual")
            axs_[i].set_ylabel("Probability of predictions")
            axs_[i].set_axis_on()
            vals, names, xs = [], [], []
            for i, col in enumerate(df_.columns):
                vals.append(df_[col].values)
                names.append(col)
                xs.append(np.random.normal(i + 1, 0.04, df_[col].values.shape[0]))
            for n, (x, val) in enumerate(zip(xs, vals)):
                scatter = boxplot.scatter(
                    x,
                    val,
                    alpha=0.9,
                    marker="o",
                    s=20,
                    facecolors="none",
                    edgecolors=[
                        "tab:blue" if x == 1 else "tab:red"
                        for x in list(data_c_.values())[n]
                    ],
                )

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Correct prediction",
                    markeredgecolor="tab:blue",
                    markerfacecolor="none",
                    markersize=5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Incorrect prediction",
                    markeredgecolor="tab:red",
                    markerfacecolor="none",
                    markersize=5,
                ),
            ]
            boxplot.legend(handles=legend_elements, loc="lower right")

        filepath = (
            output_dir
            / f"predictions_per_individual_across_study_time_{k}_{steps}_{tt}.png"
        )
        print(filepath)
        fig.tight_layout()
        fig.savefig(filepath)

        filepath = (
            output_dir
            / f"predictions_per_individual_across_study_time_box_{k}_{steps}_{tt}.png"
        )
        print(filepath)
        fig_.tight_layout()
        fig_.savefig(filepath)


def build_proba_hist(output_dir, steps, label_unhealthy, scores):
    for k in scores.keys():
        score = scores[k]
        hist_data = {}
        for label in score.keys():
            data_list = score[label]
            if len(data_list) == 0:
                continue
            label_data = []
            for elem in data_list:
                data_array = elem["test_y_pred_proba_1"]
                label_data.append(data_array)
            hist_data[label] = np.concatenate(label_data)

        plt.clf()
        plt.figure(figsize=(19.20, 10.80))
        plt.xlabel(f"Probability to be unhealthy({label_unhealthy})", size=14)
        plt.ylabel("Density", size=14)

        info = {}
        for key, value in hist_data.items():
            info[key] = hist_data[key].shape[0]

        for key, value in hist_data.items():
            plt.hist(value, density=True, bins=50, alpha=0.5, label=f"{key}")
            plt.xlim(xmin=0, xmax=1)
            plt.title(f"Histograms of prediction probabilities\n{info}")

        plt.axvline(x=0.5, color="gray", ls="--")
        plt.legend(loc="upper right")
        # plt.show()
        filename = f"histogram_of_prob_{k}_{steps}.png"
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))

        df = pd.DataFrame(hist_data.keys())
        df["equal"] = df[0].apply(lambda x: (x[-1]) == (x[0]))
        df["sup"] = df[0].apply(lambda x: (x[-1]) > (x[0]))
        df["inf"] = df[0].apply(lambda x: (x[-1]) < (x[0]))

        e = np.sum(df["equal"])
        s = np.sum(df["sup"])
        i = np.sum(df["inf"])

        max_col = np.max([e, s, i])
        max_row = 2

        plt.clf()
        fig, axs = plt.subplots(2, 1, facecolor="white", figsize=(24.0, 10.80))

        if "delmas" in str(output_dir) or "cedara" in str(output_dir): #todo use farm id
            fig, axs = plt.subplots(
                3, max_col, facecolor="white", figsize=(4.0 * max_col, 8.0)
            )
            max_row = 3

            fig.suptitle(
                f"Probability to be unhealthy({label_unhealthy})\n{info}", fontsize=14
            )
            df = df.sort_values(0, ascending=True)
            axs_ = axs.ravel()
            for ax in axs_:
                ax.set_axis_off()
            for i in range(max_row):
                for j in range(max_col):
                    a = axs[i]
                    if max_col > 1:
                        a = axs[i, j]
                    try:
                        v = df[df[["equal", "sup", "inf"][i]] == True][0].values[j]
                    except IndexError as e:
                        continue
                    a.set_ylabel("Density", size=14)
                    a.hist(hist_data[v], density=True, bins=50, alpha=1, label=f"{v}")
                    a.set_xlim(xmin=0, xmax=1)
                    a.axvline(x=0.5, color="gray", ls="--")
                    a.legend(loc="upper right")
                    a.set_axis_on()
        else:
            axs_ = axs.ravel()
            for ax in axs_:
                ax.set_axis_off()
            for i, (k, v) in enumerate(hist_data.items()):
                a = axs[i]
                a.set_ylabel("Density", size=14)
                a.hist(v, density=True, bins=50, alpha=1, label=f"{k}")
                a.set_xlim(xmin=0, xmax=1)
                a.axvline(x=0.5, color="gray", ls="--")
                a.legend(loc="upper right")
                a.set_axis_on()

        filename = f"histogram_of_prob_{k}_{steps}_grid.png"
        fig.tight_layout()
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))


def plot_histogram(x, farm_id, threshold_gap, title):
    try:
        if len(x) == 0:
            print("empty input in plot histogram!")
            return
        print("lenght=", len(x))
        print("max=", max(x))
        print("min=", min(x))
        x = pd.Series(x)

        # histogram on linear scale
        plt.subplot(211)
        plt.title(title)
        num_bins = int(max(list(set(x))))
        print("building histogram...")
        hist, bins, _ = plt.hist(x, bins=num_bins + 1, histtype="step")
        # histogram on log scale.
        # Use non-equal bin sizes, such that they look equal on log scale.
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.subplot(212)

        print("building log histogram...")
        plt.hist(x, bins=num_bins + 1, histtype="step")
        # plt.xscale('log')
        plt.yscale("log", nonposy="clip")
        print(
            "histogram_of_gap_duration_%s_%d.png"
            % (str(farm_id) + title, threshold_gap)
        )
        plt.savefig(
            "histogram_of_gap_duration_%s_%d.png"
            % (str(farm_id) + title, threshold_gap)
        )
        plt.show()
        # plt.imsave()
    except Exception as e:
        print(e)

    # num_bins = max(x)
    # fig, ax = plt.subplots()
    # # the histogram of the data
    # print("max=", max(x))
    # print("min=", min(x))
    # ax.hist(x, num_bins, density=1)
    # ax.set_xlabel('gap lenght (minutes)')
    # ax.set_ylabel('count')
    # ax.set_title('Histogram')
    # # Tweak spacing to prevent clipping of ylabel
    # fig.tight_layout()
    # plt.show()
    # print('histogram_of_gap_duration_%s_%d.png' % (farm_id, threshold_gap))
    # fig.savefig('histogram_of_gap_duration_%s_%d.png' % (farm_id, threshold_gap))


def build_report(
    output_dir,
    n_imputed_days,
    activity_days,
    data,
    y,
    steps,
    study_id,
    sampling,
    season,
    downsample,
    days,
    cv,
    cross_validation_method,
    class_healthy_label,
    class_unhealthy_label,
):
    for k, v in data.items():
        scores = {}
        report_rows_list = []
        test_precision_score0, test_precision_score1 = [], []
        test_precision_recall0, test_precision_recall1 = [], []
        test_precision_fscore0, test_precision_fscore1 = [], []
        test_precision_support0, test_precision_support1 = [], []
        test_balanced_accuracy_score = []
        aucs = []
        fit_times = []
        for item in v:
            test_precision_score0.append(item["test_precision_score_0"])
            test_precision_score1.append(item["test_precision_score_1"])
            test_precision_recall0.append(item["test_recall_0"])
            test_precision_recall1.append(item["test_recall_1"])
            test_precision_fscore0.append(item["test_fscore_0"])
            test_precision_fscore1.append(item["test_fscore_1"])
            test_precision_support0.append(item["test_support_0"])
            test_precision_support1.append(item["test_support_1"])
            fit_times.append(item["fit_time"])
            test_balanced_accuracy_score.append(item["accuracy"])
            aucs.append(item["auc"])

        scores["downsample"] = downsample
        scores["class0"] = y[y == 0].size
        scores["class1"] = y[y == 1].size
        scores["post_p"] = steps
        scores[
            "steps"
        ] = f"{study_id}->ID={n_imputed_days}->AD={activity_days}->H={str(class_healthy_label)}->UH={str(class_unhealthy_label)}->SEA={season}->{steps}->{cv}"
        scores["days"] = days
        scores["farm_id"] = study_id
        scores["balanced_accuracy_score_mean"] = np.mean(test_balanced_accuracy_score)
        scores["test_balanced_accuracy_score"] = test_balanced_accuracy_score
        scores["precision_score0_mean"] = np.mean(test_precision_score0)
        scores["test_precision_score0"] = test_precision_score0
        scores["precision_score1_mean"] = np.mean(test_precision_score1)
        scores["test_precision_score1"] = test_precision_score1
        scores["recall_score0_mean"] = np.mean(test_precision_recall0)
        scores["test_recall_score0"] = test_precision_recall0
        scores["recall_score1_mean"] = np.mean(test_precision_recall1)
        scores["test_recall_score1"] = test_precision_recall1
        scores["f1_score0_mean"] = np.mean(test_precision_recall0)
        scores["f1_score1_mean"] = np.mean(test_precision_recall1)
        scores["test_f1_score0"] = test_precision_fscore0
        scores["test_f1_score1"] = test_precision_fscore1
        scores["sampling"] = sampling
        scores["classifier"] = f"->{k}"
        scores["classifier_details"] = k
        scores["roc_auc_score_mean"] = np.mean(aucs)
        scores["roc_auc_scores"] = aucs
        scores["fit_time"] = fit_times
        report_rows_list.append(scores)

        df_report = pd.DataFrame(report_rows_list)

        df_report["class_0_label"] = str(class_healthy_label)
        df_report["class_1_label"] = str(class_unhealthy_label)
        df_report["nfold"] = cross_validation_method.get_n_splits()

        df_report["total_fit_time"] = [
            time.strftime("%H:%M:%S", time.gmtime(np.nansum(x)))
            for x in df_report["fit_time"].values
        ]

        out = output_dir / cv
        out.mkdir(parents=True, exist_ok=True)
        filename = (
            out
            / f"{k}_{activity_days}_{n_imputed_days}_{str(class_unhealthy_label)}_{study_id}_classification_report_days_{days}_{steps}_downsampled_{downsample}_sampling_{sampling}_season{season}.csv"
        )
        df_report.to_csv(filename, sep=",", index=False)
        print("filename=", filename)
        plot_ml_report(k, filename, out)


if __name__ == "__main__":
    print()
    # dir_path = "F:/Data2/job_debug/ml"
    # output_dir = "F:/Data2/job_debug/ml"
    # build_roc_mosaic(dir_path, output_dir)
