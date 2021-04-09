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
    X = X.astype(np.float)
    #step 1 find pointwise median sample [median of col1, .... median of col n].
    median_array = np.median(X, axis=0)

    #step 2 divide each sample by median array
    X_median = []
    for x in X:
        div = np.divide(x, median_array, out=np.zeros_like(x), where=median_array != 0) #return 0 if div by 0!
        X_median.append(div)

    #step 3 Within each sample (from iii) store the median value of the sample, which will produce an array of
    # median values (1 per samples).
    within_median = []
    for msample in X_median:
        within_median.append(np.median(msample))

    #step 4 Use the array of medians to scale(multiply) each original sample, which will give all quotient normalized samples.
    qnorm_sample = []
    for i, s in enumerate(X):
        qnorm_sample.append(s * within_median[i])

    #step 5 Multiply each quotient normalised sample by the total sum off all original samples divided by the sum of
    # f all element in the original corresponding sample.
    T = np.sum(X.flatten())
    t = []
    for orig_sample in X:
        t.append(np.sum(orig_sample))

    qnorm_sample_ = []
    for i, qqsample in enumerate(qnorm_sample):
        qnorm_sample_.append(qqsample * T/t[i])

    df_norm = np.array(qnorm_sample_)
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
    df = pd.DataFrame([[4, 1, 2, 2], [1, 3, 0, 3],
         [0, 7, 5, 1], [2, 0, 6, 8],
         [1, 6, 5, 4], [1, 2, 0, 4]])
    X = df.values
    print("X=", X)

    X_normalized = QuotientNormalizer().transform(X)
    # plotData(X, title="Activity sample before quotient normalisation")
    # plotData(X_normalized, title="Activity sample after quotient normalisation")

    print("after normalisation.")
    print(X_normalized)
    print("************************************")

    X = createSyntheticActivityData()
    plotData(X, title="Activity sample before quotient normalisation")

    X_normalized = QuotientNormalizer().transform(X)

    plotData(X_normalized, title="Activity sample after quotient normalisation")

