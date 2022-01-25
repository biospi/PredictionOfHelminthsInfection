import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
Utility class for static methods
"""

# def anscombe_r(value):
#     try:
#         return (value*value) / 2
#     except TypeError as e:
#         print(e)
#
# def anscombe(value):
#     try:
#         return 2 * math.sqrt(value + (3 / 8))
#     except TypeError as e:
#         print(e)

def anscombe(arr, sigma_sq=0, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha ** 2), 0)
    f = 2. * np.sqrt(v)
    return f


def inverse_anscombe(arr, sigma_sq=0, m=0, alpha=1, method='closed-form'):
    """
    Inverse of the Generalized Anscombe variance-stabilizing
    transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)


    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param m: mean of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :param method: 'closed_form' applies the closed-form approximation
    of the exact unbiased inverse. 'asym' applies the asymptotic
    approximation of the exact unbiased inverse.
    :return: inverse variance-stabilized array
    """
    sigma_sq /= alpha ** 2

    if method == 'closed-form':
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = ((arr_trunc / 2.) ** 2 + 0.25 * np.sqrt(1.5) * arr_trunc ** -1 - (11. / 8.) * arr_trunc ** -2 +
                   (5. / 8.) * np.sqrt(1.5) * arr_trunc ** -3 - (1. / 8.) - sigma_sq)
    elif method == 'asym':
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.) ** 2 - 1. / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError('Only supports the closed-form')

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse


def center_signal(y, avg):
    y_centered = y - avg
    return y_centered


def create_rec_dir(path):
    dir_path = ""
    sub_dirs = path.split("/")
    for sub_dir in sub_dirs[0:]:
        if "." in sub_dir:
            continue
        dir_path += sub_dir + "/"
        # print("sub_folder=", dir_path)
        if not os.path.exists(dir_path):
            print("mkdir", dir_path)
            os.makedirs(dir_path)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def plot_heatmap(X1, y1, X2, y2, out_dir, p1_start, p1_end, p2_start, p2_end):
    healthy_samples1 = X1[y1 == 1]
    unhealthy_samples1 = X1[y1 != 1]

    healthy_samples2 = X2[y2 == 1]
    unhealthy_samples2 = X2[y2 != 1]

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(f"healthy samples from {p1_start} to {p1_end}", f"unhealthy samples {p1_start} {p1_end}",
                        f"healthy samples {p2_start} to {p2_end}", f"unhealthy samples {p2_start} to {p2_end}"),
        y_title="",
        x_title="Time (1 min bins)",
    )

    trace1 = go.Heatmap(
        z=healthy_samples1,
        x=np.arange(0, healthy_samples1.shape[0], 1),
        y=np.arange(0, healthy_samples1.shape[1], 1),
        colorscale="Viridis",
        showscale=False
    )
    fig.append_trace(trace1, row=1, col=1)

    trace2 = go.Heatmap(
        z=unhealthy_samples1,
        x=np.arange(0, unhealthy_samples1.shape[0], 1),
        y=np.arange(0, unhealthy_samples1.shape[1], 1),
        colorscale="Viridis",
        showscale=False
    )
    fig.append_trace(trace2, row=2, col=1)

    trace3 = go.Heatmap(
        z=healthy_samples2,
        x=np.arange(0, healthy_samples2.shape[0], 1),
        y=np.arange(0, healthy_samples2.shape[1], 1),
        colorscale="Viridis",
        showscale=False
    )
    fig.append_trace(trace3, row=3, col=1)

    trace4 = go.Heatmap(
        z=unhealthy_samples2,
        x=np.arange(0, unhealthy_samples2.shape[0], 1),
        y=np.arange(0, unhealthy_samples2.shape[1], 1),
        colorscale="Viridis",
        showscale=False
    )
    fig.append_trace(trace4, row=4, col=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "samples_heatmap.html"
    output = out_dir / filename
    print(output)
    fig.write_html(str(output))


def getXY(df):
    print(df)
    X = df.iloc[:, :-1].values
    y = df["target"].values
    return X, y
