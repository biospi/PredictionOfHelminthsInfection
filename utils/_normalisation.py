import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_array
import numpy as np

np.random.seed(0)


def normalize(X):
    total_count = []
    for sample in X:
        total_count.append(np.nansum(sample))

    M = np.median(total_count)

    X_norm = []
    for i, sample in enumerate(X):
        activity = np.array(sample)
        t = total_count[i]
        a = activity * M / t
        X_norm.append(a)

    df_norm = np.array(X_norm)
    return df_norm


class QuotientNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, norm='q', *, copy=True):
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
        return normalize(X)


def createSynthetic(activity):
    pure = activity
    noise = np.random.normal(0, 1, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 3)
    synt[synt < 0] = 0
    return synt.astype(float)


def plotData(X, title="Activity sample before quotient normalisation"):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = np.array(list(range(len(sample))))
        fig.append_trace(go.Line(
            name="sample %d" % i,
            opacity=.8,
            x=timestamp,
            y=sample,
        ), row=1, col=1)

    fig.update_layout(title_text=title)
    fig.show()


def createSyntheticActivityData(n_samples=4):
    print("createSyntheticActivityData")
    samples_path = "C:/Users/fo18103/PycharmProjects/cats/src/dataset/norm_False_thresh_120/activity_cat_0_d_1_1min.csv"
    df = pd.read_csv(samples_path, header=None)
    df = df.fillna(0)
    crop = -4 - int(df.shape[1]/1.1)
    activity = df.iloc[10, : crop].values

    dataset = []
    for j in range(n_samples):
        A = createSynthetic(activity)
        dataset.append(A)

    return dataset


if __name__ == "__main__":
    print("********QuotientNormalizer*********")
    df = pd.DataFrame([[4, 1, 2, 2], [1, 3, 9, 3],
         [5, 7, 5, 1], [2, 4, 6, 8],
         [1, 6, 5, 4], [1, 2, 5, 4]])
    X = df.values
    print("X=", X)

    X_normalized = QuotientNormalizer().transform(X)
    print("after normalisation.")
    print(X_normalized)
    print("************************************")

    X = createSyntheticActivityData()
    plotData(X, title="Activity sample before quotient normalisation")

    X_normalized = QuotientNormalizer().transform(X)

    plotData(X_normalized, title="Activity sample after quotient normalisation")

