import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import pandas as pd

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
    X = df.iloc[:, :-2].values
    y = df["health"].values
    return X, y


def binarize(tagets, healty_target=1):
    return (tagets != healty_target).astype(int)


def concatenate_images(images_list, out_dir, filename="cwt_mean_per_label.png"):
    imgs = [Image.open(str(i)) for i in images_list]

    # If you're using an older version of Pillow, you might have to use .size[0] instead of .width
    # and later on, .size[1] instead of .height
    min_img_width = min(i.width for i in imgs)

    total_height = 0
    for i, img in enumerate(imgs):
        # If the image is larger than the minimum width, resize it
        if img.width > min_img_width:
            imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
        total_height += imgs[i].height

    # I have picked the mode of the first image to be generic. You may have other ideas
    # Now that we know the total height of all of the resized images, we know the height of our final image
    img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))
    y = 0
    for img in imgs:
        img_merge.paste(img, (0, y))

        y += img.height

    file_path = out_dir.parent / filename
    print(file_path)
    img_merge.save(str(file_path))


def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

def explode(df, columns):
    df['tmp'] = df.apply(lambda row: list(zip(row[columns])), axis=1)
    df = df.explode('tmp')
    df[columns] = pd.DataFrame(df['tmp'].tolist(), index=df.index)
    df.drop(columns='tmp', inplace=True)
    print(df)
    return df