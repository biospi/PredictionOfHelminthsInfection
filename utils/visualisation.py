import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from sklearn.metrics import auc

from utils.Utils import create_rec_dir, anscombe
import random
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme

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


def plot_cwt_power_sidebyside(output_samples, class_healthy_label, class_unhealthy_label, class_healthy,
                              class_unhealthy, idx_healthy, idx_unhealthy, coi_line_array, df_timedomain,
                              graph_outputdir, power_cwt_healthy, power_cwt_unhealthy, freqs, ntraces=3,
                              title="cwt_power_by_target", stepid=10, meta_size=4):
    total_healthy = df_timedomain[df_timedomain["target"] == class_healthy].shape[0]
    total_unhealthy = df_timedomain[df_timedomain["target"] == class_unhealthy].shape[0]
    plt.clf()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19.20, 7.20))
    # fig.suptitle(title, fontsize=18)

    df_healthy = df_timedomain[df_timedomain["target"] == class_healthy].iloc[:, :-meta_size].values
    df_unhealthy = df_timedomain[df_timedomain["target"] == class_unhealthy].iloc[:, :-meta_size].values

    ymin = 0
    ymax = max([np.max(df_healthy), np.max(df_unhealthy)])

    ticks = get_time_ticks(df_healthy.shape[1])

    for row in df_healthy:
        ax1.plot(ticks, row)
        ax1.set(xlabel="", ylabel="activity")
        ax1.set_title("Healthy(%s) animals %d / displaying %d" % (class_healthy_label, total_healthy, df_healthy.shape[0]))
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())

    for row in df_unhealthy:
        ax2.plot(ticks, row)
        ax2.set(xlabel="", ylabel="activity")
        ax2.set_yticks(ax2.get_yticks().tolist())
        ax2.set_xticklabels(ticks, fontsize=12)
        ax2.set_title(
            "Unhealthy(%s) animals %d / displaying %d" % (class_unhealthy_label, total_unhealthy, df_unhealthy.shape[0]))
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())

    #ax3.plot(np.log(coi_line_array), linestyle="--", linewidth=3, c="yellow")
    p = power_cwt_healthy.copy()
    p[p == -99] = np.nan
    p = np.log(p)
    ax3.imshow(p)
    ax3.set_aspect('auto')
    ax3.set_title("Healthy(%s) animals elem wise average of %d cwts" % (class_healthy_label, df_healthy.shape[0]))
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Frequency")
    # ax3.set_yscale('log')
    n_y_ticks = ax3.get_yticks().shape[0]
    labels = ["%.4f" % item for item in freqs]
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    ax3.set_yticklabels(labels_)

    #ax4.plot(coi_line_array, linestyle="--", linewidth=3, c="yellow")
    p = power_cwt_healthy.copy()
    p[p == -99] = np.nan
    p = np.log(p)
    ax4.imshow(p)
    ax4.set_aspect('auto')
    ax4.set_title("Unhealthy(%s) animals elem wise average of %d cwts" % (class_unhealthy_label, df_unhealthy.shape[0]))
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Frequency")
    #ax4.set_yscale('log')

    n_y_ticks = ax4.get_yticks().shape[0]
    labels = ["%.4f" % item for item in freqs]
    # print(labels)
    labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    ax4.set_yticklabels(labels_)

    # plt.show()
    filename = "%d_%s.png" % (stepid, title.replace(" ", "_"))
    filepath = "%s/%s" % (graph_outputdir, filename)
    # print('saving fig...')
    print(filepath)
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


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

def plotMlReport(path, output_dir):
    print("building report visualisation...")
    df = pd.read_csv(str(path), index_col=None)
    df["config"] = [format(str(x)) for x in list(zip(df.steps, df.classifier))]
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
    fig.write_html(output_dir + "/" + "ML_performance.html")
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


def plotHeatmap(X, out_dir="", title="Heatmap", filename="heatmap.html", y_log=False):
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
    fig.update_layout(xaxis_title="Time in minutes")
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
