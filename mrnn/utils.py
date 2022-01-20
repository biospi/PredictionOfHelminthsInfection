"""Utility functions for MRNN.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
--------------------------------------------------
(1) MinMaxScaler
(2) Imputation performance
"""

# Necessary packages
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import tensorflow as tf
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import time
from scipy.stats import entropy
import plotly.io as pio


def inverse_anscombe(arr, sigma_sq=0, m=0, alpha=1, method="closed-form"):
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

    if method == "closed-form":
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = (
            (arr_trunc / 2.0) ** 2
            + 0.25 * np.sqrt(1.5) * arr_trunc ** -1
            - (11.0 / 8.0) * arr_trunc ** -2
            + (5.0 / 8.0) * np.sqrt(1.5) * arr_trunc ** -3
            - (1.0 / 8.0)
            - sigma_sq
        )
    elif method == "asym":
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.0) ** 2 - 1.0 / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError("Only supports the closed-form")

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse


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
    v = np.maximum((arr / alpha) + (3.0 / 8.0) + sigma_sq / (alpha ** 2), 0)
    f = 2.0 * np.sqrt(v)
    return f


def plot_hist(data, file_name, seq_len):
    filename = Path(file_name).stem
    x = data.values.flatten()
    plt.hist(x, density=True, bins=30)  # density=False would make counts
    plt.ylabel("Probability")
    plt.xlabel("Data")
    plt.savefig(f"hist_{filename}_seql_{seq_len}.png")


def MinMaxScaler(data):
    """Normalization tool: Min Max Scaler.

    Args:
      - data: raw input data

    Returns:
      - normalized_data: minmax normalized data
      - norm_parameters: normalization parameters for rescaling if needed
    """
    min_val = np.nanmin(data, axis=0)
    data = data - min_val
    max_val = np.nanmax(data, axis=0) + 1e-8
    normalized_data = data / max_val

    norm_parameters = {"min_val": min_val, "max_val": max_val}
    return normalized_data, norm_parameters


def MinMaxScaler_(data, norm_parameters):
    """Normalization tool: Min Max Scaler.

    Args:
      - data: raw input data

    Returns:
      - normalized_data: minmax normalized data
      - norm_parameters: normalization parameters for rescaling if needed
    """
    min_val = norm_parameters["min_val"]
    data = data - min_val
    max_val = norm_parameters["max_val"]
    normalized_data = data / max_val
    return normalized_data


def remove_empty_col(item):
    i_list = []
    for i in range(item.shape[1]):
        v = item[:, i]
        if len(v[v > 0]) > int(item.shape[0]/4):
            i_list.append(i)

    return i_list


def initialise_time_matrix(data):
    matrix = data.copy()
    if len(data.shape) == 2:
        for k in range(matrix.shape[1]):
            matrix[:, k] = np.arange(1, matrix.shape[0] + 1)
    else:
        for l in range(matrix.shape[0]):
            for k in range(matrix.shape[2]):
                matrix[l, :, k] = np.arange(1, matrix.shape[1] + 1)
    return matrix


def find_dense(iteration, data, mask, tm, args, features, streams, run_id, n, w_str):

    idx_f = np.where(np.in1d(features, np.array(streams)))[0]
    data_ = data[:, :, idx_f]
    mask_ = mask[:, :, idx_f]
    tm_ = tm[:, :, idx_f]
    features_f = features[idx_f]

    # list = []
    # for i in range(data_.shape[0]):
    #     item = data_[i]
    #     z_values_count = item[item == 0].shape[0]
    #     list.append(z_values_count)

    #idxs = np.array(list).argsort()[:n]
    idxs = np.arange(data_.shape[0])

    data_f, data_f_visu = [], []
    mask_f, mask_f_visu = [], []
    tm_f, tm_f_visu = [], []

    #indices = None
    for item, item_m, item_tm in zip(data_[idxs], mask_[idxs], tm_[idxs]):
        # if indices is None:
        #     indices = remove_empty_col(item)
        if item[item == 0].shape == item.flatten().shape:
            #print("skip sample!")
            continue

        if args.filter_training:
            is_valid = True
            for j in range(item.shape[1]):
                elem = item[:, j]
                c = np.sum((elem > 0))
                print(c)
                if c <= 100:
                    is_valid = False
                    break

            if not is_valid:
                continue
                # item[:] = 0
                # item_m[:] = 0
                # item_tm[:] = initialise_time_matrix(np.empty(item.shape))

        data_f.append(item)
        mask_f.append(item_m)
        tm_f.append(item_tm)

        pad = np.ones(item.shape)
        pad[:] = np.nan
        pad = pad[0:int(pad.shape[0]/10), :]
        data_f_visu.append(item)
        data_f_visu.append(pad)
        mask_f_visu.append(item_m)
        mask_f_visu.append(pad)
        tm_f_visu.append(item_tm)
        tm_f_visu.append(pad)

    df = pd.DataFrame(np.vstack(data_f_visu), columns=features_f)
    df_m = pd.DataFrame(np.vstack(mask_f_visu), columns=features_f)
    df_t = pd.DataFrame(np.vstack(tm_f_visu), columns=features_f)

    #n = len(data_f)
    n = idxs.shape[0]
    print(f"Number of windows assigned is {n}")

    #data = data[['40101310013', '40101310040', '40101310109', '40101310110']]

    plot_heatmap(
        streams,
        run_id,
        [df, df_m, df_t],
        n,
        args.file_name,
        args.missing_rate,
        args.seq_len,
        args.h_dim,
        iteration,
        args.learning_rate,
        args.batch_size,
        f"training_input_nwindows_{n}",
        w_str,
        0,
        0,
        inptut_title=True
    )
    data_f, mask_f, tm_f = np.array(data_f), np.array(mask_f), np.array(tm_f)

    return data_f, mask_f, tm_f, n, features_f


def Denormalization(X, norm_parameters):
    min = norm_parameters["min_val"]
    max = norm_parameters["max_val"]
    renormalized_data = X * max
    renormalized_data = renormalized_data + min
    # enormalized_data = np.round(data, 2)
    return renormalized_data


def plot_loss_curve(
    run_id,
    streams,
    n,
    start_i,
    end_i,
    data_list,
    file,
    title,
    seql,
    missing_rate,
    iter,
    learning_rate,
    has_weather,
    labels=None,
):

    plt.clf()
    cm = plt.get_cmap("gist_rainbow")
    fig = plt.figure(figsize=(19.80, 7.20))
    ax = fig.add_subplot(111)
    NUM_COLORS = len(data_list)
    LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]
    NUM_STYLES = len(LINE_STYLES)

    for i, data in enumerate(data_list):
        if labels is not None:
            lines = ax.plot(data, label=labels[i])
        else:
            lines = ax.plot(data)

        lines[0].set_color(cm(i // NUM_STYLES * float(NUM_STYLES) / NUM_COLORS))
        lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_xscale("log")
    ax.legend(
        loc="upper right", fancybox=True, shadow=False, ncol=5, fontsize="x-small"
    )
    ax.set_title(
        f"{title} dataset:{file} \n sequence length:{seql} missing_rate:{missing_rate} iteration:{iter} learning_rate:{learning_rate}"
    )
    filename = f"{run_id}_{file}_{title.replace(' ','_')}_{start_i}_{end_i}_seql_{seql}_{n}_missr_{missing_rate}_iter_{iter}_lr_{learning_rate}_hw_{has_weather}.png"
    out_dir = Path("loss_curve") / f"{run_id}_missingrate_{missing_rate}_seql_{seql}_iteration_{iter}_hw_{has_weather}_n_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


def concat_html(figs, filename, titles):

    fig = make_subplots(
        subplot_titles = tuple(titles),
        rows=len(figs),
        cols=1,
        y_title="",
        x_title="Time (1 min bins)",
    )

    for i, f in enumerate(figs):
        fig.append_trace(f, row=i + 1, col=1)

    fig.update_layout(
        height=100*len(figs)
    )
    fig.write_html(filename)
    #fig.write_image(filename.replace(".html", ".png"))
    print(filename)


def merge_mrnn_steps_html_figures(label, i, run_id, miss_rate, seq_len, iteration, weather_str, n):
    #return
    print("merge_mrnn_steps_html_figures...")
    out_dir = Path(
        "mrnn_heatmaps") / f"{run_id}_missingrate_{miss_rate}_seql_{seq_len}_iteration_{iteration}_hw_{weather_str}_n_{n}"
    files = out_dir.glob("*.json")
    figs = []
    titles = []
    v_max = []
    v_min = []
    for f in files:
        if f.stem[0] == str(i) and label in f.stem:
            print(f)
            with open(str(f), 'r') as f_fig:
                fig = pio.from_json(f_fig.read())
                figs.append(fig["data"][0])
                # z = np.array(fig["data"][0].z, dtype=np.float)
                # v_min.append(np.nanmin(z))
                # v_max.append(np.nanmax(z))
                titles.append(f.stem)
    #concat_html(figs, f"{str(out_dir)}/{i}_{label}.html", titles)


def plot_heatmap_imstep(streams, input, title, label, run_id, miss_rate, seq_len, iteration, weather_str, n, id=0, m=None, i=0):
    #return
    #print("plot_heatmap_imstep...")

    matrix = input.copy()

    if m is not None:
        matrix[m == 0] = np.nan

    matrix = np.vstack(matrix)
    xaxix_label = [x for x in range(matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=matrix[0:1440*31, :].T,
        x=xaxix_label,
        y=streams,
        # zmin=0,
        # zmax=1,
        colorscale="Viridis",
        showscale=False
        )
    )
    fig.update_layout(
        title=title)

    out_dir = Path("mrnn_heatmaps") / f"{run_id}_missingrate_{miss_rate}_seql_{seq_len}_iteration_{iteration}_hw_{weather_str}_n_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{i}_{id}_{label}_{title}.json"
    #fig.write_json(str(output))
    print(output)

    # fig = go.Figure()
    # for i in range(matrix.shape[1]):
    #     row = matrix[0:1440*2, i]
    #     fig.add_trace(go.Scatter(x=np.arange(row.shape[0]), y=row,
    #                              mode='lines+markers',
    #                              name=streams[i]))
    # #fig.show()
    # output = out_dir / f"{i}_{id}_{label}_{title}_lines.html"
    # fig.write_html(str(output))
    # print(output)


def plot_heatmap(
    streams,
    run_id,
    dfs,
    n,
    file_name,
    miss_rate,
    seq_len,
    h_dim,
    iteration,
    learning_rate,
    batch_size,
    title_,
    weather_str,
    start_i,
    end_i,
    slice=133920,
    inptut_title = False,
    add_title = False
):
    ifilename = Path(file_name).stem

    t1 = f"MRNN before imputation Heatmap \n n samples:{n} mr:{miss_rate} seql:{seq_len} h_dim:{h_dim} iteration:{iteration} learning_rate:{learning_rate} batch_size:{batch_size} has_weather:{weather_str}"
    t2 = f"MRNN after imputation Heatmap \n mr:{miss_rate} seql:{seq_len} h_dim:{h_dim} iteration:{iteration} learning_rate:{learning_rate} batch_size:{batch_size} has_weather:{weather_str}"
    t3 = f"After linear imputation Heatmap \n mr:{miss_rate} seql:{seq_len} h_dim:{h_dim} iteration:{iteration} learning_rate:{learning_rate} batch_size:{batch_size} has_weather:{weather_str}"

    if inptut_title:
        t1 = f"data n samples:{n}"
        t2 = "mask"
        t3 = "time_matrix"

    if slice > dfs[0].shape[0]:
        n_fig = np.array([0, dfs[0].shape[0]])
    else:
        n_fig = np.arange(0, dfs[0].shape[0], slice)

    figs = []
    for k in range(n_fig.shape[0] - 1):
        start = n_fig[k]
        end = n_fig[k + 1]

        fig = make_subplots(
            rows=len(dfs),
            cols=1,
            subplot_titles=(t1, t2, t3),
            y_title="",
            x_title="Time (1 min bins)",
        )
        for i, df in enumerate(dfs):
            df = df.iloc[start:end, :]
            # plt.savefig(output)
            xaxix_label = np.arange(0, df.shape[0], 1)
            yaxis_label = [f"_{x}" for x in df.columns]
            # yaxis_label = np.arange(0, len(df.columns), 1)
            matrix = df.T.values
            matrix[0][0] = 0 #prevent plotly axis bug when matrix only contains nan
            trace = go.Heatmap(
                z=matrix,
                x=xaxix_label,
                y=yaxis_label,
                #colorbar=dict(x=1 + i / 30, title=f"fig:{i}"),
                colorscale="Viridis",
                showscale=False
            )
            figs.append(trace)
            fig.append_trace(trace, row=i + 1, col=1)
        filename = f"{run_id}_{title_}_heatmap_{ifilename}_{k}_{start}_{end}_missingrate_{miss_rate}_{start_i}_{end_i}_seql_{seq_len}_h_dim_{h_dim}_iteration_{iteration}_learning_rate_{learning_rate}_batch_size_{batch_size}_hw_{weather_str}.html"
        filename_concat = f"{run_id}_{title_}_heatmap_{ifilename}_concat_{k}_{start}_{end}_missingrate_{miss_rate}_{start_i}_{end_i}_seql_{seq_len}_h_dim_{h_dim}_iteration_{iteration}_learning_rate_{learning_rate}_batch_size_{batch_size}_hw_{weather_str}.html"
        out_dir = Path("heatmaps") / f"{run_id}_missingrate_{miss_rate}_seql_{seq_len}_iteration_{iteration}_hw_{weather_str}_n_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / filename
        # if k in [0, n_fig.shape[0] - 2]:
        print(output)
        fig.write_html(str(output))

    concat_html(figs, str(out_dir / filename_concat), [t1, t2, t3])


def linear_interpolate(a):
    df = pd.DataFrame(a).interpolate(method="linear", axis=0).fillna(0)
    return df.values


def stream_info(file):
    df = pd.read_csv(file)
    stream_names = df.columns
    data = df.values
    pos_count_list = []
    for i in range(data.shape[1]):
        stream = data[:, i]
        pos_count = len(stream[stream > 0])
        pos_count_list.append([pos_count, stream_names[i]])

    pos_count_list.sort(key=lambda x: x[0])
    pos_count_list = pos_count_list[::-1]
    print(pos_count_list)
    print("streams sorted by number of positive values:")
    s_list = []
    for s in pos_count_list:
        print(s[1], s[0])
        s_list.append(s[1])
    print(s_list)


def rmse(ori_x_rmse, imputed_x, data_mask):
    performances = []
    for k in np.arange(0, 1.0, 0.1):
        d_m = data_mask.copy()
        d_m[ori_x_rmse > k] = 0
        diff_ = ori_x_rmse * d_m - imputed_x * d_m
        nominator_ = np.sum(diff_ ** 2)
        denominator_ = np.sum(d_m)
        #print("nominator=", nominator_)
        #print("denominator=", denominator_)
        if denominator_ != 0:
            performance = np.sqrt(float(nominator_) / float(denominator_))
        else:
            performance = np.nan
        performances.append(performance)
    print(performances)
    return performances


def imputation_performance(
    streams,
    run_id,
    n,
    data_mask,
    ori_x,
    imputed_x,
    m,
    metric_name,
    features,
    args,
    dataset_name,
    norm_parameters,
    weather_str,
    start_i,
    end_i,
    iteration,
    timestamp,
    date_str,
    export_csv
):
    """Performance metrics for imputation.

    Args:
      - ori_x: original complete data (without missing values)
      - imputed_x: imputed data from incomplete data
      - m: observation indicator
      - metric_name: mae, mse, or rmse

    Returns:
      - performance: imputation performance in terms or mae, mse, or rmse
    """

    assert metric_name in ["mae", "mse", "rmse"]

    no, seq_len, dim = ori_x.shape

    # Reshape 3d array to 2d array
    imputed_x = np.reshape(imputed_x, [no * seq_len, dim])
    ori_x = np.reshape(ori_x, [no * seq_len, dim])
    m = np.reshape(m, [no * seq_len, dim])

    ori_x[m == 0] = np.nan

    # i_idx = np.arange(0, ori_x.shape[0], seq_len).tolist()
    # # before last window last idx
    # i_idx[-1] = i_idx[-1] - 1
    # # last window
    # w_idx = np.arange(i_idx[-1] + 1, ori_x.shape[0]).tolist()
    # idx = i_idx + w_idx

    # ori_x = ori_x[idx]
    # imputed_x = imputed_x[idx]
    # m = m[idx]

    # Only compute the imputation performance if m = 0 (missing)
    imputed_li_x = linear_interpolate(ori_x)
    performance_li = np.nan
    performance = np.nan
    if metric_name == "mae":
        performance = mean_absolute_error(ori_x, imputed_x, 1 - m)
    elif metric_name == "mse":
        performance = mean_squared_error(ori_x, imputed_x, 1 - m)
    elif metric_name == "rmse":
        if args.missing_rate == 0:
            print("no missingness for rmse evaluation. set to nan.")
        else:
            if args.missing_rate != 0:
                # ori_x_rmse = ori_x.copy()
                # ori_x_rmse[np.isnan(ori_x_rmse)] = 0
                # # data_mask[ori_x_rmse == 0] = 0
                #
                # performance = rmse(ori_x_rmse, imputed_x, data_mask[:imputed_x.shape[0], :])
                # performance_li = rmse(ori_x_rmse, imputed_li_x, data_mask[:imputed_x.shape[0], :])
                #
                # # performance = np.sqrt(mean_squared_error(ori_x_rmse, imputed_x, data_mask))
                # # performance_li = np.sqrt(mean_squared_error(ori_x_rmse, imputed_li_x, data_mask))
                print(f"performance: {performance} performance_li:{performance_li}")

    imputed_li_x = imputed_li_x[::-1]
    imputed_x = imputed_x[::-1]
    ori_x = ori_x[::-1]

    imputed_feature = pd.DataFrame(imputed_x, columns=features)
    imputed_li_feature = pd.DataFrame(imputed_li_x, columns=features)
    data_o = pd.DataFrame(ori_x, columns=features)

    print(data_o.shape, imputed_feature.shape)
    plot_heatmap(
        streams,
        run_id,
        [data_o, imputed_feature, imputed_li_feature],
        n,
        args.file_name,
        args.missing_rate,
        args.seq_len,
        args.h_dim,
        iteration,
        args.learning_rate,
        args.batch_size,
        "imputed",
        weather_str,
        start_i,
        end_i,
        slice= 1440 * 7,
        add_title=True
    )

    # filename = f"{run_id}_imputed_data_{dataset_name}_{start_i}_{end_i}_seql_{args.seq_len}_{n}_mr_{args.missing_rate}_iter_{iteration}_lr_{args.learning_rate}_hw_{weather_str}.csv"
    # output_dir = Path("imputed_data") / f"{run_id}_missingrate_{args.missing_rate}_seql_{args.seq_len}_iteration_{iteration}_hw_{weather_str}_n_{n}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # filepath = output_dir / filename
    # print(filepath)
    # result.to_csv(str(filepath), sep=",", index=False)
    ori_x_ = Denormalization(data_o, norm_parameters)
    ori_x_ = np.exp(ori_x_)
    ori_x_ = np.around(inverse_anscombe(ori_x_))

    imputed_x_ = Denormalization(imputed_feature, norm_parameters)
    imputed_x_ = np.exp(imputed_x_)
    imputed_x_ = np.around(inverse_anscombe(imputed_x_))

    imputed_li_x_ = Denormalization(imputed_li_feature, norm_parameters)
    imputed_li_x_ = np.exp(imputed_li_x_)
    imputed_li_x_ = np.around(inverse_anscombe(imputed_li_x_))

    if True:
        output_dir = Path(
            "imputed_data") / f"{run_id}_missingrate_{args.missing_rate}_seql_{args.seq_len}_iteration_{iteration}_hw_{weather_str}_n_{n}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for j in range(ori_x.shape[1]):
            id = features[j]
            df_ = pd.DataFrame(columns=["timestamp", "date_str", "first_sensor_value", "first_sensor_value_mrnn", "first_sensor_value_li", "imputed"])
            df_["timestamp"] = timestamp
            df_["date_str"] = date_str
            df_["first_sensor_value"] = ori_x_[id]
            df_["first_sensor_value_mrnn"] = imputed_x_[id]
            df_["first_sensor_value_li"] = imputed_li_x_[id]
            df_["imputed"] = np.isnan(ori_x_[id]).astype(int)
            filepath = output_dir / f"{id}.csv"
            print(filepath)
            df_.to_csv(filepath, index=False)

    return performance, performance_li, output_dir


def plot_model_struct(model, filename="model.png"):
    tf.keras.utils.plot_model(
        model,
        to_file=filename,
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
        layer_range=None,
    )


def plot_rmse_performance(start_i, end_i, n, run_id):
    files = Path("performance").glob("*.csv")
    dfs = []
    for file in files:
        # if int(file.name[0]) != 8:
        #     continue
        df = pd.read_csv(file)
        if df["iteration"][0] != 50:
            continue
        dfs.append(df)

    df_ = pd.concat(dfs, axis=0)
    #df_ = df_.fillna(4)
    df_[["has_weather"]] = df_[["has_weather"]].fillna(value="no_weather")
    xaxis = "seq_len"
    df_ = df_.sort_values(xaxis)

    list_of_df = [
        g
        for _, g in df_.groupby(
            ["run_id", "dataset_name", "seq_len", "missing_rate", "learning_rate", "has_weather", "n_samples"]
        )
    ]

    plt.clf()
    plt.figure(figsize=(19.80, 7.20))
    for i, d in enumerate(list_of_df):
        # xaxis = "missing_rate"
        n_streams = d["n_streams"].values[0]
        run_id = d["run_id"].values[0]
        dataset = d["dataset_name"].values[0]
        seql = d["seq_len"].values[0]
        missing_rate = d["missing_rate"].values[0]
        learning_rate = d["learning_rate"].values[0]
        iteration = d["iteration"].values[0]
        has_weather = d["has_weather"].values[0]
        x = d[xaxis].values
        y = d["performance"].values
        y = np.array([np.array([float(x.strip(' []')) for x in y[n].split(',')]) for n in range(d["performance"].shape[0])])
        y_li = d["performance_li"].values
        y_li = np.array([np.array([float(x.strip(' []')) for x in y_li[n].split(',')]) for n in range(d["performance"].shape[0])])
        plt.plot(x, y[:, 5], label=f"RMSE({0.5:.2f}) mrnn imputation n_streams:{n_streams} seql:{seql}", marker="x")
        print(y[:, 5])
        if i == 0:
            plt.plot(x, y_li[:, 5], label=f"RMSE linear imputation n_streams:{n_streams}", marker="x")

    plt.xlabel(xaxis)
    #plt.xscale("log")
    plt.ylabel("RMSE")
    plt.gca().legend(loc="upper right")


    plt.title(
        f"RMSE performance dataset:{dataset} \n number of streams:{n_streams} sequence length:{seql} iteration:{iteration} learning_rate:{learning_rate} has_weather:{has_weather}"
    )
    filename = f"{run_id}_rmse_performance_{dataset}_{xaxis}_{has_weather}_{start_i}_{end_i}_seql_{seql}_n_{n}_missr_{missing_rate}_iter_{iteration}_lr_{learning_rate}.png"
    out_dir = Path("performance")
    filepath = out_dir / filename
    print(filepath)
    plt.savefig(filepath)


        # plt.clf()
        # plt.figure(figsize=(19.80, 7.20))
        # x = d[xaxis].values
        # f_t = d["fit_time"].values
        # i_t = d["impute_time"].values
        # plt.plot(x, f_t, label="Fit time", marker="x")
        # plt.plot(x, i_t, label="Imputation time", marker="x")
        # plt.xlabel(xaxis)
        # plt.xscale("log")
        # plt.ylabel("Time")
        #
        # formatter = matplotlib.ticker.FuncFormatter(
        #     lambda s, f_t: time.strftime("%H:%M:%S", time.gmtime(s))
        # )
        # plt.gca().yaxis.set_major_formatter(formatter)
        # plt.gca().legend(loc="upper right")
        #
        # plt.title(
        #     f"Computation time dataset:{dataset} \n number of streams:{n_streams}  sequence length:{seql} iteration:{iteration} learning_rate:{learning_rate} has_weather:{has_weather}"
        # )
        # filename = f"{run_id}_time_performance_{dataset}_{xaxis}_{has_weather}_{start_i}_{end_i}_seql_{seql}_n_{n}_missr_{missing_rate}_iter_{iteration}_lr_{learning_rate}.png"
        # out_dir = Path("performance")
        # filepath = out_dir / filename
        # print(filepath)
        # plt.savefig(filepath)
