import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
from datetime import datetime, timedelta

from utils.Utils import create_rec_dir, anscombe

np.random.seed(0)


def normalize(X, out_dir):
    out_dir_ = out_dir + "_normalisation"
    traces = []
    X = X.astype(np.float)
    zmin, zmax = np.min(np.log(anscombe(X))), np.max(np.log(anscombe(X)))

    traces.append(plotHeatmap(zmin, zmax, np.array(X).copy(), out_dir_, "STEP 0 | Samples", "0_X_samples.html", y_log=True))

    #step 1 find pointwise median sample [median of col1, .... median of col n].
    median_array = np.median(X, axis=0)
    traces.append(plotLine([median_array], out_dir_, "STEP 1 | find pointwise median sample [median of col1, .... median of col n]", "1_median_array.html"))

    #step 2 divide each sample by median array keep div by 0 as NaN!!
    X_median = []
    for x in X:
        div = np.divide(x, median_array)
        div[div == -np.inf] = np.nan
        div[div == np.inf] = np.nan
        div[div == 0] = np.nan
        X_median.append(div)
    traces.append(plotHeatmap(zmin, zmax, np.array(X_median).copy(), out_dir_, "STEP 2 | divide each sample by median array "
                                                                        "keep div by 0 as NaN, set 0 to NaN", "2_X_median.html", y_log=True))

    #step 3 Within each sample (from iii) store the median value of the sample(excluding 0 value!), which will produce an array of
    # median values (1 per samples).
    within_median = []
    for msample in X_median:
        clean_sample = msample[~np.isnan(msample)]
        within_median.append(np.median(clean_sample))
    traces.append(plotLine([within_median], out_dir_, "STEP 3 | Within each sample (rows from step2) store the median"
                                                      " value of the sample, which will produce an array of median "
                                                      "values (1 per samples)", "3_within_median.html", x_axis_count=True, y_log=True))

    #step 4 Use the array of medians to scale(divide) each original sample, which will give all quotient normalized samples.
    qnorm_samples = []
    for i, s in enumerate(X):
        qnorm_samples.append(np.divide(s, within_median[i]))
    traces.append(plotHeatmap(zmin, zmax, np.array(qnorm_samples).copy(), out_dir_, "STEP 4 | Use the array of medians"
                                                                             " to scale(divide) each original sample,"
                                                                             " which will give all quotient normalized samples.",
                              "4_qnorm_sample.html", y_log=True))

    #step 5 substract step 1 from step 4
    diff = X - np.array(qnorm_samples)
    traces.append(plotHeatmap(np.min(diff), np.max(diff), diff, out_dir_, "STEP 5 | Substract step 1 (original samples)"
                                                                          " from step 4 (quotient normalised samples)", "5_diff.html"))

    plot_all(traces, out_dir_, title="Quotient Normalisation 5 STEPS")

    df_norm = np.array(qnorm_samples)
    return df_norm


class QuotientNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, norm='q', *, out_dir=None, copy=True):
        self.out_dir = out_dir
        self.norm = norm
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
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        #copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr')
        norm = normalize(X, self.out_dir)
        # norm_simple = normalize_simple(X, self.out_dir)
        return norm


def createSynthetic(activity):
    pure = activity
    noise = np.random.normal(0, 200, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 1.5)
    synt[synt < 0] = 0
    return synt.astype(int)


def plot_all(traces, out_dir, title="Quotient Normalisation STEPS", filename="steps.html", simple= False):
    ts = []
    for trace in traces:
        ts.append(trace[1])
    cbarlocs = [.96, 0.75, .60, 0.50, 0.23, .05]
    if simple:
        cbarlocs = [.96, 0, .50, 0, .05]
    fig = make_subplots(rows=len(traces), cols=1, subplot_titles=tuple(ts))
    for i, trace in enumerate(traces):
        trace[0].colorbar = dict(len=0.10, y=cbarlocs[i])
        fig.append_trace(trace[0], row=i+1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=False)
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)


def plotLine(X, out_dir="", title="title", filename="file.html", x_axis_count=False, y_log=False):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = get_time_ticks(len(sample))
        if x_axis_count:
            timestamp = list(range(len(timestamp)))
        if y_log:
            sample_log = np.log(sample)
        trace = go.Line(
            opacity=.8,
            x=timestamp,
            y=sample_log if y_log else sample,
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


def plotHeatmap(zmin, zmax, X, out_dir="", title="Heatmap", filename="heatmap.html", y_log=False):
    # fig = make_subplots(rows=len(transponders), cols=1)
    ticks = get_time_ticks(X.shape[1])
    fig = make_subplots(rows=1, cols=1)
    if y_log:
        X_log = np.log(anscombe(X))
    trace = go.Heatmap(
            z=X_log if y_log else X,
            x=ticks,
            y=list(range(X.shape[0])),
            zmin=zmin,
            zmax=zmax,
            colorscale='Viridis')
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    #fig.show()
    create_rec_dir(out_dir)
    file_path = out_dir + "/" + filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


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


if __name__ == "__main__":
    print("********QuotientNormalizer*********")
    df = pd.DataFrame([[4, 1, 2, 2], [1, 3, 0, 3],
         [0, 7, 5, 1], [2, 0, 6, 8],
         [1, 6, 5, 4], [1, 2, 0, 4]])
    X = df.values
    print("X=", X)

    out_dir = "F:/Data2/_normalisation_1"
    X_normalized = QuotientNormalizer(out_dir=out_dir).transform(X)
    # plotData(X, title="Activity sample before quotient normalisation")
    # plotData(X_normalized, title="Activity sample after quotient normalisation")

    print("after normalisation.")
    print(X_normalized)
    print("************************************")

    out_dir = "F:/Data2/_normalisation_2"
    X = createSyntheticActivityData()
    plotLine(X, out_dir=out_dir, title="Activity sample before quotient normalisation")

    X_normalized = QuotientNormalizer(out_dir=out_dir).transform(X)

    plotLine(X_normalized, out_dir=out_dir, title="Activity sample after quotient normalisation")
    print()

