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

from utils.Utils import create_rec_dir


def plot_cwt_power(out_dir, i, activity, power_masked, coi_line_array, freqs):
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(19.20, 7.20))
    fig.suptitle("Signal , CWT", fontsize=18)

    ticks = get_time_ticks(len(activity))
    axs[0].plot(ticks, activity)
    axs[0].set_title("Time domain signal")
    axs[0].set(xlabel="Time", ylabel="activity")
    # axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # axs[0].xaxis.set_major_locator(mdates.DayLocator())
    with np.errstate(invalid='ignore'):  # ignore numpy divide by zero warning
        axs[1].imshow(np.log(power_masked))
    if(len(coi_line_array) > 0):
        axs[1].plot(coi_line_array, linestyle="--", linewidth=5, c="white")
    axs[1].set_aspect('auto')
    axs[1].set_title("CWT")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Frequency of wavelet")
    # axs[1].set_yscale('log')
    # n_x_ticks = axs[1].get_xticks().shape[0]
    # labels = [item.strftime("%H:%M") for item in ticks]
    # labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_x_ticks)))]
    # labels_[0:2] = labels[0]
    # labels_[-2:] = labels[0]
    # axs[1].set_xticklabels(labels_)
    #
    # n_y_ticks = axs[1].get_yticks().shape[0]
    # labels = ["%.4f" % item for item in freqs]
    # # print(labels)
    # labels_ = np.array(labels)[list(range(1, len(labels), int(len(labels) / n_y_ticks)))]
    # axs[1].set_yticklabels(labels_)
    # plt.show()

    filename = "%d_cwt.png" % i
    filepath = "%s/%s" % (out_dir, filename)
    create_rec_dir(filepath)
    # print('saving fig...')
    fig.savefig(filepath)
    # print("saved!")
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
                cwt[i:, j] = -99
                coi_line.append(i)
                break

    return cwt, coi_line


def compute_cwt(X, out_dir):
    print("compute_cwt...")
    out_dir = out_dir + "_cwt"
    plotHeatmap(X, out_dir=out_dir, title="Time domain samples", force_xrange=True, filename="time_domain_samples.html")

    cwt = []
    i = 0
    for activity in tqdm(X):
        y = activity
        w = wavelet.Morlet()
        coefs, scales, freqs, coi, _, _ = wavelet.cwt(y, 1, wavelet=w)
        coefs_cc = np.conj(coefs)
        with np.errstate(divide='ignore'):#ignore numpy divide by zero warning
            #power_cwt = np.log(np.real(np.multiply(coefs, coefs_cc)))
            power_cwt = np.real(np.multiply(coefs, coefs_cc))

        # power_cwt[power_cwt == -np.inf] = 0  # todo check why inf output
        power_masked, coi_line_array = mask_cwt(power_cwt.copy(), coi, scales)
        #power_masked, coi_line_array = power_cwt, []

        plot_cwt_power(out_dir, i, activity, power_masked, coi_line_array, freqs)
        power_flatten_masked = np.array(power_masked.flatten())
        power_flatten_masked = power_flatten_masked[power_flatten_masked != -99]
        cwt.append(power_flatten_masked)
        i += 1
    cwt = np.array(cwt)

    plotHeatmap(cwt, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT.html", head=True)
    return cwt


class CWT(TransformerMixin, BaseEstimator):
    def __init__(self, *, out_dir=None, copy=True):
        self.out_dir = out_dir
        self.copy = copy

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
        cwt = compute_cwt(X, self.out_dir)
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
    return trace, title


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def plotHeatmap(X, out_dir="", title="Heatmap", filename="heatmap.html", force_xrange=False, head=False):
    # fig = make_subplots(rows=len(transponders), cols=1)
    if head:
        X = X[:2, :]
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
    #fig.show()
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


if __name__ == "__main__":
    print("********CWT*********")

    out_dir = "F:/Data2/_cwt_debug"
    X = np.array(createSyntheticActivityData())
    plotHeatmap(X, out_dir=out_dir, title="Activity samples", filename="X.html")

    X_CWT = CWT(out_dir=out_dir).transform(X)

    plotHeatmap(X_CWT, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT.html")
    print("********END*********")

